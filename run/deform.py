import sys
import os
import pathlib
sys.path.append(os.path.join(pathlib.Path(__file__).parent.absolute(), '..'))
import glob
import copy
import torch
import numpy as np
import time
from options import TrainOptions
from data import create_data_loader, create_dataset
from models import create_model
from models.mvs.mvs_points_model import MvsPointsModel
from models.mvs import mvs_utils, filter_utils
from models.dynamic_point_field.model import deform_cloud, Siren, RayBender
from models.dynamic_point_field.utils import *
import matplotlib.pyplot as plt
from pprint import pprint
from utils.visualizer import Visualizer
from utils import format as fmt
from run.evaluate import report_metrics
torch.manual_seed(0)
np.random.seed(0)
import random
import cv2
from PIL import Image
from tqdm import tqdm
import gc
import point_cloud_utils as pcu

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def mse2psnr(x): return -10.* torch.log(x)/np.log(10.)

def save_image(img_array, filepath):
    assert len(img_array.shape) == 2 or (len(img_array.shape) == 3
                                         and img_array.shape[2] in [3, 4])

    if img_array.dtype != np.uint8:
        img_array = (np.clip(img_array, 0, 1) * 255).astype(np.uint8)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    Image.fromarray(img_array).save(filepath)


def nearest_view(campos, raydir, xyz, id_list):
    cam_ind = torch.zeros([0,1], device=campos.device, dtype=torch.long)
    step=10000
    for i in range(0, len(xyz), step):
        dists = xyz[i:min(len(xyz),i+step), None, :] - campos[None, ...] # N, M, 3
        dists_norm = torch.norm(dists, dim=-1) # N, M
        dists_dir = dists / (dists_norm[...,None]+1e-6) # N, M, 3
        dists = dists_norm / 200 + (1.1 - torch.sum(dists_dir * raydir[None, :],dim=-1)) # N, M
        cam_ind = torch.cat([cam_ind, torch.argmin(dists, dim=1).view(-1,1)], dim=0) # N, 1
    return cam_ind



def masking(mask, firstdim_lst, seconddim_lst):
    first_lst = [item[mask, ...] if item is not None else None for item in firstdim_lst]
    second_lst = [item[:, mask, ...] if item is not None else None for item in seconddim_lst]
    return first_lst, second_lst



def render_vid(model, dataset, visualizer, opt, bg_info, steps=0, gen_vid=True):
    print('-----------------------------------Rendering-----------------------------------')
    model.eval()
    total_num = dataset.total
    print("test set size {}, interval {}".format(total_num, opt.test_num_step))
    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size

    height = dataset.height
    width = dataset.width
    visualizer.reset()
    for i in range(0, total_num):
        data = dataset.get_dummyrot_item(i)
        raydir = data['raydir'].clone()
        pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
        # cam_posts.append(data['campos'])
        # cam_dirs.append(data['raydir'] + data['campos'][None,...])
        # continue
        visuals = None
        stime = time.time()

        for k in range(0, height * width, chunk_size):
            start = k
            end = min([k + chunk_size, height * width])
            data['raydir'] = raydir[:, start:end, :]
            data["pixel_idx"] = pixel_idx[:, start:end, :]
            # print("tmpgts", tmpgts["gt_image"].shape)
            # print(data["pixel_idx"])
            model.set_input(data)
            if opt.bgmodel.endswith("plane"):
                img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, fg_masks, bg_ray_lst = bg_info
                if len(bg_ray_lst) > 0:
                    bg_ray_all = bg_ray_lst[data["id"]]
                    bg_idx = data["pixel_idx"].view(-1,2)
                    bg_ray = bg_ray_all[:, bg_idx[:,1].long(), bg_idx[:,0].long(), :]
                else:
                    xyz_world_sect_plane = mvs_utils.gen_bg_points(data)
                    bg_ray, _ = model.set_bg(xyz_world_sect_plane, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, data["plane_color"], fg_masks=fg_masks, vis=visualizer)
                data["bg_ray"] = bg_ray

            model.test()
            curr_visuals = model.get_current_visuals(data=data)
            if visuals is None:
                visuals = {}
                for key, value in curr_visuals.items():
                    if key == "gt_image": continue
                    chunk = value.cpu().numpy()
                    visuals[key] = np.zeros((height * width, 3)).astype(chunk.dtype)
                    visuals[key][start:end, :] = chunk
            else:
                for key, value in curr_visuals.items():
                    if key == "gt_image": continue
                    visuals[key][start:end, :] = value.cpu().numpy()

        for key, value in visuals.items():
            visualizer.print_details("{}:{}".format(key, visuals[key].shape))
            visuals[key] = visuals[key].reshape(height, width, 3)
        print("num.{} in {} cases: time used: {} s".format(i, total_num // opt.test_num_step, time.time() - stime), " at ", visualizer.image_dir)
        visualizer.display_current_results(visuals, i)

    # visualizer.save_neural_points(200, np.concatenate(cam_posts, axis=0),None, None, save_ref=False)
    # visualizer.save_neural_points(200, np.concatenate(cam_dirs, axis=0),None, None, save_ref=False)
    # print("vis")
    # exit()

    print('--------------------------------Finish Evaluation--------------------------------')
    if gen_vid:
        del dataset
        visualizer.gen_video("coarse_raycolor", range(0, total_num), 0)
        print('--------------------------------Finish generating vid--------------------------------')

    return



def test(xsrc, vsrc, kp_idxs, model, dataset, visualizer, opt, bg_info, test_steps=0, gen_vid=True, lpips=True):
    print('-----------------------------------Testing-----------------------------------')
    print('device: ', device)
    model.eval()
    total_num = dataset.total
    print("test set size {}, interval {}".format(total_num, opt.test_num_step)) # 1 if test_steps == 10000 else opt.test_num_step
    patch_size = opt.random_sample_size
    chunk_size = patch_size * patch_size

    height = dataset.height
    width = dataset.width
    visualizer.reset()
    count = 0
    for i in range(0, total_num, opt.test_num_step): # 1 if test_steps == 10000 else opt.test_num_step
        # deform points
        keypoint_dir = os.path.join(opt.data_root, opt.scan, "keypoint")
        # target keypoint
        vtrg = pcu.load_mesh_v(f"{keypoint_dir}/{i}.obj")
        vtrg_total_num = vtrg.shape[0]
        if kp_idxs is not None:
            # sample keypoint
            vtrg = vtrg[kp_idxs]
        # draw pcd
        draw_pcd = True
        if draw_pcd and opt.sample_num > 0:
            # define colors for verts
            vrgb = torch.zeros([len(vtrg), 4]).to(device)
            vrgb[:, 0] = 0
            vrgb[:, 1] = 0.5
            vrgb[:, 2] = 0
            vrgb[:, 3] = 1
            if vtrg.shape[0] < 50:
                radius = 0.02
            elif vtrg.shape[0] > 500:
                radius = 0.005
            else:
                radius = 0.01
            yvsrc = render_points(vsrc, vrgb, azim=0, radius=radius, image_size=1024).detach().cpu().numpy()
            yvtrg = render_points(vtrg, vrgb, azim=0, radius=radius, image_size=1024).detach().cpu().numpy()
            plt.axis('off')
            plt.imshow(yvsrc)
            plt.savefig(os.path.join(opt.checkpoints_dir + opt.name, f"{i}_{vtrg.shape[0]}_{vtrg_total_num}_{opt.sample_num}_vsrc.png"))
            plt.close()
            plt.axis('off')
            plt.imshow(yvtrg)
            plt.savefig(os.path.join(opt.checkpoints_dir + opt.name, f"{i}_{vtrg.shape[0]}_{vtrg_total_num}_{opt.sample_num}_vtrg.png"))
            plt.close()
            # cv2.imwrite(os.path.join(opt.checkpoints_dir + opt.name, f"{vtrg.shape[0]}_{vtrg_total_num}_{opt.sample_num}_vsrc.png"), yvsrc)
            # cv2.imwrite(os.path.join(opt.checkpoints_dir + opt.name, f"{vtrg.shape[0]}_{vtrg_total_num}_{opt.sample_num}_vtrg.png"), yvtrg)
        xpred = deform(xsrc, vsrc, vtrg)
        if opt.ray_bend == 1:
            model.raybender.set_trg(xpred)
        # coordinate transform
        xsrc_x, xsrc_y, xsrc_z = xpred[:, 0], xpred[:, 1], xpred[:, 2]
        xpred = torch.stack([xsrc_x, -xsrc_z, xsrc_y], dim=-1)
        
        # print
        print(fmt.RED + f'Deform image {i}, Used keypoint {vtrg.shape[0]}, Total keypoint {vtrg_total_num}, Total pointcloud {xsrc.shape[0]}' + fmt.END)

        # breakpoint()
        with torch.no_grad():
            model.neural_points.xyz = torch.nn.Parameter(xpred)
            # model.neural_points.points_color = nn.Parameter(saved_features["neural_points.points_color"])
            data = dataset.get_item(i)
            raydir = data['raydir'].clone()
            pixel_idx = data['pixel_idx'].view(data['pixel_idx'].shape[0], -1, data['pixel_idx'].shape[3]).clone()
            edge_mask = torch.zeros([height, width], dtype=torch.bool)
            edge_mask[pixel_idx[0,...,1].to(torch.long), pixel_idx[0,...,0].to(torch.long)] = 1
            edge_mask=edge_mask.reshape(-1) > 0
            np_edge_mask=edge_mask.numpy().astype(bool)
            totalpixel = pixel_idx.shape[1]
            tmpgts = {}
            tmpgts["gt_image"] = data['gt_image'].clone()
            tmpgts["gt_mask"] = data['gt_mask'].clone() if "gt_mask" in data else None
            # print("data['gt_image']")
            # data.pop('gt_image', None)
            data.pop('gt_mask', None)

            visuals = None
            stime = time.time()
            ray_masks = []
            for k in range(0, totalpixel, chunk_size):
                start = k
                end = min([k + chunk_size, totalpixel])
                data['raydir'] = raydir[:, start:end, :]
                data["pixel_idx"] = pixel_idx[:, start:end, :]
                model.set_input(data)

                if opt.bgmodel.endswith("plane"):
                    img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, fg_masks, bg_ray_lst = bg_info
                    if len(bg_ray_lst) > 0:
                        bg_ray_all = bg_ray_lst[data["id"]]
                        bg_idx = data["pixel_idx"].view(-1,2)
                        bg_ray = bg_ray_all[:, bg_idx[:,1].long(), bg_idx[:,0].long(), :]
                    else:
                        xyz_world_sect_plane = mvs_utils.gen_bg_points(data)
                        bg_ray, _ = model.set_bg(xyz_world_sect_plane, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, data["plane_color"], fg_masks=fg_masks, vis=visualizer)
                    data["bg_ray"] = bg_ray

                model.test()
                curr_visuals = model.get_current_visuals(data=data)
                chunk_pixel_id = data["pixel_idx"].cpu().numpy().astype(np.int32)
                if visuals is None:
                    visuals = {}
                    for key, value in curr_visuals.items():
                        if value is None or key=="gt_image":
                            continue
                        chunk = value.cpu().numpy()
                        visuals[key] = np.zeros((height, width, 3)).astype(chunk.dtype)
                        visuals[key][chunk_pixel_id[0,...,1], chunk_pixel_id[0,...,0], :] = chunk
                else:
                    for key, value in curr_visuals.items():
                        if value is None or key=="gt_image":
                            continue
                        visuals[key][chunk_pixel_id[0,...,1], chunk_pixel_id[0,...,0], :] = value.cpu().numpy()
                if "ray_mask" in model.output and "ray_masked_coarse_raycolor" in opt.test_color_loss_items:
                    ray_masks.append(model.output["ray_mask"] > 0)
            if len(ray_masks) > 0:
                ray_masks = torch.cat(ray_masks, dim=1)
            gt_image = torch.zeros((height*width, 3), dtype=torch.float32)
            gt_image[edge_mask, :] = tmpgts['gt_image'].clone()
            if 'gt_image' in model.visual_names:
                visuals['gt_image'] = gt_image
            if 'gt_mask' in curr_visuals:
                visuals['gt_mask'] = np.zeros((height, width, 3)).astype(chunk.dtype)
                visuals['gt_mask'][np_edge_mask,:] = tmpgts['gt_mask']
            if 'ray_masked_coarse_raycolor' in model.visual_names:
                visuals['ray_masked_coarse_raycolor'] = np.copy(visuals["coarse_raycolor"]).reshape(height, width, 3)
                print(visuals['ray_masked_coarse_raycolor'].shape, ray_masks.cpu().numpy().shape)
                visuals['ray_masked_coarse_raycolor'][ray_masks.view(height, width).cpu().numpy() <= 0,:] = 0.0
            if 'ray_depth_masked_coarse_raycolor' in model.visual_names:
                visuals['ray_depth_masked_coarse_raycolor'] = np.copy(visuals["coarse_raycolor"]).reshape(height, width, 3)
                visuals['ray_depth_masked_coarse_raycolor'][model.output["ray_depth_mask"][0].cpu().numpy() <= 0] = 0.0
            if 'ray_depth_masked_gt_image' in model.visual_names:
                visuals['ray_depth_masked_gt_image'] = np.copy(tmpgts['gt_image']).reshape(height, width, 3)
                visuals['ray_depth_masked_gt_image'][model.output["ray_depth_mask"][0].cpu().numpy() <= 0] = 0.0
            if 'gt_image_ray_masked' in model.visual_names:
                visuals['gt_image_ray_masked'] = np.copy(tmpgts['gt_image']).reshape(height, width, 3)
                visuals['gt_image_ray_masked'][ray_masks.view(height, width).cpu().numpy() <= 0,:] = 0.0
            for key, value in visuals.items():
                if key in opt.visual_items:
                    visualizer.print_details("{}:{}".format(key, visuals[key].shape))
                    visuals[key] = visuals[key].reshape(height, width, 3)


            print("num.{} in {} cases: time used: {} s".format(i, total_num // opt.test_num_step, time.time() - stime), " at ", visualizer.image_dir)
            visualizer.display_current_results(visuals, i, opt=opt)

            acc_dict = {}
            if "coarse_raycolor" in opt.test_color_loss_items:
                loss = torch.nn.MSELoss().to("cuda")(torch.as_tensor(visuals["coarse_raycolor"], device="cuda").view(1, -1, 3), gt_image.view(1, -1, 3).cuda())
                acc_dict.update({"coarse_raycolor": loss})
                print("coarse_raycolor", loss, mse2psnr(loss))

            if "ray_mask" in model.output and "ray_masked_coarse_raycolor" in opt.test_color_loss_items:
                masked_gt = tmpgts["gt_image"].view(1, -1, 3).cuda()[ray_masks,:].reshape(1, -1, 3)
                ray_masked_coarse_raycolor = torch.as_tensor(visuals["coarse_raycolor"], device="cuda").view(1, -1, 3)[:,edge_mask,:][ray_masks,:].reshape(1, -1, 3)
                loss = torch.nn.MSELoss().to("cuda")(ray_masked_coarse_raycolor, masked_gt)
                acc_dict.update({"ray_masked_coarse_raycolor": loss})
                visualizer.print_details("{} loss:{}, PSNR:{}".format("ray_masked_coarse_raycolor", loss, mse2psnr(loss)))

            if "ray_depth_mask" in model.output and "ray_depth_masked_coarse_raycolor" in opt.test_color_loss_items:
                ray_depth_masks = model.output["ray_depth_mask"].reshape(model.output["ray_depth_mask"].shape[0], -1)
                masked_gt = torch.masked_select(tmpgts["gt_image"].view(1, -1, 3).cuda(), (ray_depth_masks[..., None].expand(-1, -1, 3)).reshape(1, -1, 3))
                ray_depth_masked_coarse_raycolor = torch.masked_select(torch.as_tensor(visuals["coarse_raycolor"], device="cuda").view(1, -1, 3), ray_depth_masks[..., None].expand(-1, -1, 3).reshape(1, -1, 3))

                loss = torch.nn.MSELoss().to("cuda")(ray_depth_masked_coarse_raycolor, masked_gt)
                acc_dict.update({"ray_depth_masked_coarse_raycolor": loss})
                visualizer.print_details("{} loss:{}, PSNR:{}".format("ray_depth_masked_coarse_raycolor", loss, mse2psnr(loss)))
            print(acc_dict.items())
            visualizer.accumulate_losses(acc_dict)
            count+=1

    visualizer.print_losses(count)
    psnr = visualizer.get_psnr(opt.test_color_loss_items[0])
    print('--------------------------------Finish Test Rendering--------------------------------')
    report_metrics(visualizer.image_dir, visualizer.image_dir, visualizer.image_dir, ["psnr", "lpips", "vgglpips", "rmse"] if lpips else ["psnr", "ssim", "rmse"], [i for i in range(0, total_num, opt.test_num_step)], imgStr="step-%04d-{}.png".format(opt.visual_items[0]),gtStr="step-%04d-{}.png".format(opt.visual_items[1]))
    print('--------------------------------Finish Evaluation--------------------------------')
    if gen_vid:
        del dataset
        visualizer.gen_video("coarse_raycolor", range(0, total_num, opt.test_num_step), test_steps)
        print('--------------------------------Finish generating vid--------------------------------')
    return psnr


def get_latest_epoch(resume_dir):
    os.makedirs(resume_dir, exist_ok=True)
    str_epoch = [file.split("_")[0] for file in os.listdir(resume_dir) if file.endswith("_states.pth")]
    int_epoch = [int(i) for i in str_epoch]
    return None if len(int_epoch) == 0 else str_epoch[int_epoch.index(max(int_epoch))]


def main():
    torch.backends.cudnn.benchmark = True
    opt = TrainOptions().parse()
    # cur_device = torch.device('cuda:{}'.format(opt.gpu_ids[0]) if opt.
    #                           gpu_ids else torch.device('cpu'))
    print("opt.color_loss_items ", opt.color_loss_items)

    if opt.debug:
        torch.autograd.set_detect_anomaly(True)
        print(fmt.RED +
              '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        print('Deformation')
        print(
            '++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++' +
            fmt.END)
    # initialize deformation
    xsrc, vsrc, vrgb, kp_idxs = init_deformation(opt)
    raybender = RayBender(xsrc, 8)
    visualizer = Visualizer(opt)
    train_dataset = create_dataset(opt)
    img_lst=None
    with torch.no_grad():
        print(opt.checkpoints_dir + opt.name + "/*_net_ray_marching.pth")
        if opt.bgmodel.endswith("plane"):
            _, _, _, _, _, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst = gen_points_filter_embeddings(train_dataset, visualizer, opt)

        resume_dir = os.path.join(opt.checkpoints_dir, opt.name)
        if opt.resume_iter == "best":
            opt.resume_iter = "latest"
        resume_iter = opt.resume_iter if opt.resume_iter != "latest" else get_latest_epoch(resume_dir)
        if resume_iter is None:
            visualizer.print_details("No previous checkpoints at iter {} !!", resume_iter)
            exit()
        else:
            opt.resume_iter = resume_iter
            visualizer.print_details('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
            visualizer.print_details('test at {} iters'.format(opt.resume_iter))
            visualizer.print_details(f"Iter: {resume_iter}")
            visualizer.print_details('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
        opt.mode = 2
        opt.load_points=1
        opt.resume_dir=resume_dir
        opt.resume_iter = resume_iter
        opt.is_train=True

    model = create_model(opt)
    model.setup(opt, train_len=len(train_dataset))
    # breakpoint()
    if opt.ray_bend == 0:
        raybender = None
        print('--------------------------No Bending!------------------------------------')
    model.set_raybender(raybender)
    # create test loader
    test_opt = copy.deepcopy(opt)
    test_opt.is_train = False
    test_opt.random_sample = 'no_crop'
    test_opt.random_sample_size = min(48, opt.random_sample_size)
    test_opt.batch_size = 1
    test_opt.n_threads = 0
    test_opt.prob = 0
    test_opt.split = "test"
    visualizer.reset()

    fg_masks = None
    test_bg_info = None
    if opt.bgmodel.endswith("plane"):
        test_dataset = create_dataset(test_opt)
        bg_ray_test_lst = create_all_bg(test_dataset, model, img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst)
        test_bg_info = [img_lst, c2ws_lst, w2cs_lst, intrinsics_all, HDWD_lst, fg_masks, bg_ray_test_lst]
        del test_dataset
        # if opt.vid > 0:
        #     render_dataset = create_render_dataset(test_opt, opt, resume_iter, test_num_step=opt.test_num_step)
    ############ initial test ###############
    # with torch.no_grad():
    test_opt.nerf_splits = ["test"]
    test_opt.split = "test"
    test_opt.name = opt.name + "/test_{}".format(resume_iter)
    test_opt.test_num_step = opt.test_num_step
    test_dataset = create_dataset(test_opt)
    model.opt.is_train = 0
    model.opt.no_loss = 1
    test(xsrc, vsrc, kp_idxs, model, test_dataset, Visualizer(test_opt), test_opt, test_bg_info, test_steps=resume_iter)

def sample_kp(kp, num=-1):    # num: sample number or ratio
    if num > 0:
        if num > 1:
            kp_idxs = np.random.choice(kp.shape[0], int(num)).tolist()
        elif num <= 1:
            kp_idxs = np.random.choice(kp.shape[0], int(kp.shape[0] * num)).tolist()
    else:
        kp_idxs = None

    return kp_idxs

def init_deformation(opt):
    # get vsrc_idx
    vsrc_idx = int(np.loadtxt(os.path.join(opt.data_root, opt.scan, "src_id.txt")))
    # load static NeRF pointcloud
    saved_features = torch.load(opt.checkpoints_dir + opt.name + "/" + str(opt.resume_iter) + "_net_ray_marching.pth", map_location=device)
    keypoint_dir = os.path.join(opt.data_root, opt.scan, "keypoint")
    point_cloud = saved_features["neural_points.xyz"].cpu().numpy()
    # source point cloud & keypoint
    vsrc = pcu.load_mesh_v(f"{keypoint_dir}/{vsrc_idx}.obj")
    print(f"vsrc size: {vsrc.shape[0]}")
    xsrc, csrc = point_cloud[:, :3], point_cloud[:, 3:]
    if opt.sample_num > 0:
        print(f"sample num: {opt.sample_num}")
        # sample keypoint
        kp_idxs = sample_kp(vsrc, opt.sample_num)
        vsrc = vsrc[kp_idxs]
    else:
        kp_idxs = None
    # to tensor
    xsrc = torch.Tensor(xsrc).to(device)
    csrc = torch.Tensor(csrc / 255).to(device)
    vsrc = torch.Tensor(vsrc).to(device)
    # coordinate transform
    xsrc_x, xsrc_y, xsrc_z = xsrc[:, 0], xsrc[:, 1], xsrc[:, 2]
    xsrc = torch.stack([xsrc_x, xsrc_z, -xsrc_y], dim=-1)
    # define colors for vertices
    vrgb = torch.zeros([len(vsrc), 4]).to(device)
    vrgb[:, 1] = 1
    vrgb[:, 3] = 1
    return xsrc, vsrc, vrgb, kp_idxs

def deform(xsrc, vsrc, vtrg):
    # to tensor
    vtrg = torch.Tensor(vtrg).to(device)
    # xtrg = torch.Tensor(xtrg).to(device)
    # ctrg = torch.Tensor(ctrg / 255).to(device)

    # run model
    deform_model = Siren(in_features=3,
                  hidden_features=128,
                  hidden_layers=3,
                  out_features=3, outermost_linear=True,
                  first_omega_0=30, hidden_omega_0=30.).to(device).train()

    deform_cloud(deform_model,
                 xsrc=xsrc, vsrc=vsrc, vtrg=vtrg,
                 init_lr=1.0e-4,
                 n_steps=2000,
                 use_chamfer=False,
                 use_guidance=True,
                 use_isometric=True,
                 guided_weight=1.0e4,
                 isometric_weight=1.0e3)
    deform_model.eval()
    xpred = xsrc + deform_model(xsrc).detach().clone()
    xpred = torch.Tensor(xpred).to(device)
    return xpred

# def main():
#     device = 'cuda' if torch.cuda.is_available() else 'cpu'
#     ROOT_DIR = os.getcwd()
#
#     # dirs
#     # checkpoint_dir = os.path.join(ROOT_DIR, "/checkpoints/nerfsynth/human_cuda/")
#     # data_dir = os.path.join(ROOT_DIR, "/data_src/nerf/nerf_synthetic/human")
#     # deformed_checkpoint_dir = os.path.join(ROOT_DIR, "deformed")
#     # if not os.path.exists(deformed_checkpoint_dir):
#     #     os.makedirs(deformed_checkpoint_dir)
#
#     # checkpoint_filename = "80000_net_ray_marching.pth"
#     vsrc_idx = 67
#     keypoint_dir = os.path.join(opt.data_root, opt.scan, "keypoint")
#
#     # =====================================
#     # run dynamic point field deformation
#     # =====================================
#
#
#
#
#     # read targets
#     target_list = glob.glob(os.path.join(keypoint_dir, "*.obj"))
#     for target in target_list:
#         print(target)
#         # target keypoint
#         vtrg = pcu.load_mesh_v(target)
#         # to tensor
#         vtrg = torch.Tensor(vtrg).to(device)
#         # xtrg = torch.Tensor(xtrg).to(device)
#         # ctrg = torch.Tensor(ctrg / 255).to(device)
#
#         # run model
#         model = Siren(in_features=3,
#                       hidden_features=128,
#                       hidden_layers=3,
#                       out_features=3, outermost_linear=True,
#                       first_omega_0=30, hidden_omega_0=30.).to(device).train()
#
#         deform_cloud(model,
#                      xsrc=xsrc, vsrc=vsrc, vtrg=vtrg,
#                      init_lr=1.0e-4,
#                      n_steps=2000,
#                      use_chamfer=False,
#                      use_guidance=True,
#                      use_isometric=True,
#                      guided_weight=1.0e4,
#                      isometric_weight=1.0e3)
#         model.eval()
#         xpred = xsrc + model(xsrc).detach().clone()
#         xpred = torch.Tensor(xpred).to(device)
#
#         # coordinate transform
#         xsrc_x, xsrc_y, xsrc_z = xpred[:, 0], xpred[:, 1], xpred[:, 2]
#         xpred = torch.stack([xsrc_x, -xsrc_z, xsrc_y], dim=-1)
#
#         # save to checkpoint
#         saved_features["neural_points.xyz"] = xpred
#         save_dir = os.path.join(deformed_checkpoint_dir, os.path.basename(target).split('.')[0])
#         if not os.path.exists(save_dir):
#             os.makedirs(save_dir)
#         torch.save(saved_features, os.path.join(save_dir, checkpoint_filename))
#
#         # =====================================
#         # run point-nerf rendering
#         # =====================================
#         print("start point-nerf rendering\n")
#         # 1. initialize
#         opt = parse_options()
#         opt.checkpoints_dir = deformed_checkpoint_dir
#         opt.name = os.path.basename(target).split('.')[1]
#         # cur_device = torch.device('cuda:{}'.format(opt.gpu_ids[0]) if opt.
#         #                           gpu_ids else torch.device('cpu'))
#         # print("opt.color_loss_items ", opt.color_loss_items)
#
#         # if opt.debug:
#         #     torch.autograd.set_detect_anomaly(True)
#         #     print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#         #     print('TESTING')
#         #     print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
#         visualizer = Visualizer(opt)
#
#         model, test_opt = initalize_pointnerf_model(opt)
#         visualizer.reset()
#         fg_masks = None
#         test_bg_info = None
#         with torch.no_grad():
#             test_dataset = create_dataset(test_opt)
#             model.opt.is_train = 0
#             model.opt.no_loss = 1
#             test(model, test_dataset, Visualizer(test_opt), test_opt, test_bg_info, test_steps=test_opt.resume_iter)


if __name__ == '__main__':
    main()
