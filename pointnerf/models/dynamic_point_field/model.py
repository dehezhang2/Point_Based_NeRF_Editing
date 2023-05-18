import sys
import torch
from torch import nn
import numpy as np
from tqdm import tqdm
from pytorch3d.ops.knn import knn_points
from pytorch3d.loss.chamfer import chamfer_distance
from pytorch3d.transforms import matrix_to_quaternion, quaternion_to_matrix
from torch.nn.functional import l1_loss
from collections import OrderedDict
import open3d as o3d
from sklearn.neighbors import NearestNeighbors

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SineLayer(nn.Module):
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.

    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a
    # hyperparameter.

    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)

    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first

        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)

        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                            1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                            np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False,
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)

            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                             np.sqrt(6 / hidden_features) / hidden_omega_0)

            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords  # .requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output  # , coords

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)

                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()

                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else:
                x = layer(x)

                if retain_grad:
                    x.retain_grad()

            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations

class RayBender:
    def __init__(self, src_dir, k=8):
        self.src_tri_mesh = o3d.io.read_triangle_mesh(src_dir)
        self.src_tri_mesh.compute_adjacency_list()
        self.src_adj = self.src_tri_mesh.adjacency_list
        self.vsrc, self.fsrc = np.asarray(self.src_tri_mesh.vertices), np.asarray(self.src_tri_mesh.triangles)
        self.k = 8
    def set_trg(self, trg_dir):
        # compute the rotation matrix for each vertex
        trg_tri_mesh = o3d.io.read_triangle_mesh(trg_dir)
        vtrg = np.asarray(trg_tri_mesh.vertices)
        trg_tri_mesh.compute_adjacency_list()
        trg_adj = trg_tri_mesh.adjacency_list
        self.rotation_per_vertex = []
        print("===setting up the KD tree===")
        for i in tqdm(range(min(len(self.src_adj), len(trg_adj)))):
            joint_adj = self.src_adj[i].union(trg_adj[i])
            joint_adj = np.asarray(list(joint_adj))
            joint_adj_mask = np.logical_and(joint_adj < self.vsrc.shape[0] , joint_adj < vtrg.shape[0])
            joint_adj= joint_adj[joint_adj_mask]

            X = self.vsrc[joint_adj]
            X = X - self.vsrc[i]

            Y = vtrg[joint_adj]
            Y = Y - vtrg[i]
            u, _, vh = np.linalg.svd(X.T @ Y)
            # right multiply R => X R = Y
            R = u @ vh
            self.rotation_per_vertex.append(R)
        self.rotation_per_vertex = np.array(self.rotation_per_vertex)
        self.kd_tree = NearestNeighbors(n_neighbors=self.k, algorithm='kd_tree').fit(vtrg)
    
    def query(self, query_points):
        #input: query_points: B, R, SR, 3
        #output: rotations: B, R, SR, 3, 3
        q_shape = query_points.shape
        query_points = query_points.view(-1, 3).cpu().numpy()
        distances, indices = self.kd_tree.kneighbors(query_points)
        # N * K
        distances = torch.Tensor(distances).to(device=device)
        # N * K * 4
        rotations = matrix_to_quaternion(torch.Tensor(self.rotation_per_vertex[indices]).to(device=device))
        weight = 1. / torch.clamp(distances, min= 1e-6)
        weight = weight / torch.clamp(torch.sum(weight, dim=-1, keepdim=True), min=1e-8)
        rotation_Nlerp = torch.einsum('nkd,nk->nd', rotations, weight)
        rotation_Nlerp = rotation_Nlerp / rotation_Nlerp.norm(dim=-1)[:, None]
        # N * K * 3 * 3
        rotation_Nlerp = quaternion_to_matrix(rotation_Nlerp)
        rotation_Nlerp = rotation_Nlerp.reshape((q_shape[0], q_shape[1], q_shape[2], 3, 3))
        return rotation_Nlerp
    
    def bend_rays(self, query_points, query_dirs):
        #input: query_points: B, R, SR, 3
        #input: query_dirs: B, R, SR, 3
        #output: new_dirs: B, R, SR, 3
        sample_rot = self.query(query_points)
        return torch.einsum("brsa,brsca->brsc", query_dirs, sample_rot)
        
def isometric_loss(x_canonical, x_deformed, n_neighbors=5):
    """
    Computes the isometric loss between two sets of points, which measures the discrepancy
    between their pairwise distances.

    Parameters
    ----------
    x_canonical : array-like, shape (n_points, n_dims)
        The canonical (reference) point set, where `n_points` is the number of points
        and `n_dims` is the number of dimensions.
    x_deformed : array-like, shape (n_points, n_dims)
        The deformed (transformed) point set, which should have the same shape as `x_canonical`.
    n_neighbors : int, optional
        The number of nearest neighbors to use for computing pairwise distances.
        Default is 5.

    Returns
    -------
    loss : float
        The isometric loss between `x_canonical` and `x_deformed`, computed as the L1 norm
        of the difference between their pairwise distances. The loss is a scalar value.
    Raises
    ------
    ValueError
        If `x_canonical` and `x_deformed` have different shapes.
    """

    if x_canonical.shape != x_deformed.shape:
        raise ValueError("Input point sets must have the same shape.")

    _, nn_ix, _ = knn_points(x_canonical.unsqueeze(0),
                             x_canonical.unsqueeze(0),
                             K=n_neighbors,
                             return_sorted=True)

    dists_canonical = torch.cdist(x_canonical[nn_ix], x_canonical[nn_ix])
    dists_deformed = torch.cdist(x_deformed[nn_ix], x_deformed[nn_ix])

    loss = l1_loss(dists_canonical, dists_deformed)

    return loss

# guided learning
def deform_cloud(model, xsrc, xtrg=None, vsrc=None, vtrg=None,
                 use_isometric=True, use_isometric_inter=True,
                 use_chamfer=True, use_guidance=True,
                 n_chamfer_samples=10 ** 4, n_steps=20000, init_lr=1.0e-4,
                 isometric_weight=1.0e4, isometric_inter_max=1.0,
                 guided_weight=1.0e4, chamfer_weight=1.0e4,
                 iso_n_neighbors=5, eval_every_nth_step=100):
    """
    Deform a point cloud using a neural network model.

    Parameters
    ----------
    model : torch.nn.Module
        The neural network model to use for deformation.
    xsrc : numpy.ndarray
        The source point cloud to deform.
    xtrg : numpy.ndarray, optional
        The target point cloud to match (used in chamfer distance loss).
    vsrc : numpy.ndarray, optional
        The source keypoints to use as guidance for deformation.
    vtrg : numpy.ndarray, optional
        The target keypoints to match (used in guided loss).
    use_isometric : bool, optional
        Whether to use isometric loss (default is True).
    use_isometric_inter : bool, optional
        Whether to use isometric loss between halfway frames (default is True).
    use_chamfer : bool, optional
        Whether to use chamfer distance loss (default is True).
    use_guidance : bool, optional
        Whether to use guided loss (default is True).
    n_chamfer_samples : int, optional
        The number of points to sample for chamfer distance loss (default is 10**4).
    n_steps : int, optional
        The number of optimization steps (default is 20000).
    init_lr : float, optional
        The initial learning rate for the optimizer (default is 1.0e-4).
    isometric_weight : float, optional
        The weight for isometric loss (default is 1.0e4).
    isometric_inter_max : float, optional
        The interval for isometric loss between frames (default is 1.0). Check Section 3.2 for details
    guided_weight : float, optional
        The weight for guided loss (default is 1.0e4).
    chamfer_weight : float, optional
        The weight for chamfer distance loss (default is 1.0e4).
    iso_n_neighbors : int, optional
        The number of neighbors to use for isometric loss (default is 5).
    eval_every_nth_step : int, optional
        The number of steps between evaluations (default is 100).

    """

    model = model.train()
    l1_loss = torch.nn.L1Loss()
    optm = torch.optim.Adam(model.parameters(), lr=init_lr)
    schedm = torch.optim.lr_scheduler.ReduceLROnPlateau(optm, verbose=True, patience=1)

    if use_chamfer and xtrg is None:
        print("no target cloud provided, ignoring...")
        use_chamfer = False

    if use_guidance and vtrg is None:
        print("no keypoints provided, ignoring...")
        use_guidance = False

    guided_loss_total = 0
    chamfer_loss_total = 0
    isometric_loss_total = 0
    total_loss = 0
    n_r = 0

    for i in range(0, n_steps):

        if use_isometric or use_chamfer or use_isometric_inter:
            xbatch_src = torch.Tensor(xsrc[np.random.choice(len(xsrc), n_chamfer_samples, replace=False)]).to(device)
            xbatch_deformed = xbatch_src + model(xbatch_src)

        if use_isometric_inter:
            # additionally enforce isometry between frames
            t = isometric_inter_max * np.random.uniform()
            xbatch_deformed_halfway = xbatch_src + isometric_inter_max * model(xbatch_src)

        loss = 0

        if use_isometric:
            iso_loss = isometric_weight * isometric_loss(xbatch_src, xbatch_deformed, n_neighbors=iso_n_neighbors)
            loss += iso_loss
            isometric_loss_total += float(iso_loss)

        if use_isometric_inter:
            iso_loss += isometric_weight * isometric_loss(xbatch_src, xbatch_deformed_halfway,
                                                          n_neighbors=iso_n_neighbors)
            loss += iso_loss
            isometric_loss_total += float(iso_loss)

        if use_guidance:
            vsrc_deformed = vsrc + model(vsrc)
            guidance_loss = guided_weight * l1_loss(vsrc_deformed, vtrg)
            loss += guidance_loss
            guided_loss_total += float(guidance_loss)

        if use_chamfer:
            xbatch_trg = torch.Tensor(xtrg[np.random.choice(len(xtrg), n_chamfer_samples, replace=False)]).to(device)
            chamfer_loss = chamfer_weight * chamfer_distance(xbatch_deformed.unsqueeze(0),
                                                             xbatch_trg.unsqueeze(0))[0]
            loss += chamfer_loss
            chamfer_loss_total += float(chamfer_loss)

        total_loss += float(loss)
        n_r += 1

        optm.zero_grad()
        loss.backward()
        optm.step()

        if i % eval_every_nth_step == 0:
            guided_loss_total /= n_r
            isometric_loss_total /= n_r
            chamfer_loss_total /= n_r
            total_loss /= n_r

            schedm.step(float(total_loss))

            print("%05d tloss: %03f, guidl: %03f, chaml. :%03f, isol: %03f"
                  % (i, total_loss, guided_loss_total, chamfer_loss_total, isometric_loss_total))

            guided_loss_total = 0
            chamfer_loss_total = 0
            isometric_loss_total = 0
            total_loss = 0
            n_r = 0

    guided_loss_total /= n_r
    isometric_loss_total /= n_r
    chamfer_loss_total /= n_r
    total_loss /= n_r

    print("%05d tloss: %03f, guidl: %03f, chaml. :%03f, isol: %03f"
          % (i, total_loss, guided_loss_total, chamfer_loss_total, isometric_loss_total))

    return