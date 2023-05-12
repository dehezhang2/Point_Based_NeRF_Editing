import sys
import torch
import torch
from pytorch3d.structures import Pointclouds
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVOrthographicCameras,
    PerspectiveCameras,
    PointsRasterizationSettings,
    PulsarPointsRenderer,
    PointsRenderer,
    SfMPerspectiveCameras,
    PulsarPointsRenderer,
    PerspectiveCameras,
    PointsRasterizer,
    AlphaCompositor,
    NormWeightedCompositor
)
from pytorch3d.structures import Meshes

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def normalise_verts(V, V_scale=None, V_center=None):
    # Normalize mesh

    if V_scale is not None and V_center is not None:

        V = V - V_center
        V *= V_scale

    else:

        V_max, _ = torch.max(V, dim=0)
        V_min, _ = torch.min(V, dim=0)
        V_center = (V_max + V_min) / 2.
        V = V - V_center

        # Find the max distance to origin
        max_dist = torch.sqrt(torch.max(torch.sum(V ** 2, dim=-1)))
        V_scale = (1. / max_dist)
        V *= V_scale

    return V, V_scale, V_center


def normals_to_rgb(n):
    return torch.abs(n * 0.5 + 0.5)


def get_point_renderer(image_size, radius=0.05, points_per_pixel=50):
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
        points_per_pixel=points_per_pixel
    )

    rasterizer = PointsRasterizer(cameras=FoVOrthographicCameras(),
                                  raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=(0, 0, 0, 0))
    )

    return renderer


def get_camera(dist=1, elev=0, azim=0):
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)

    cam = PerspectiveCameras(R=R, T=T)

    return cam


def render_points(x, xf, dist=1, elev=0, azim=0, image_size=512, radius=0.01, points_per_pixel=50, scale_val=1.0):
    x = x.to(device)
    xf = xf.to(device)
    renderer = get_point_renderer(image_size=image_size, radius=radius, points_per_pixel=points_per_pixel)
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cam = FoVOrthographicCameras(R=R, T=T, scale_xyz=((scale_val, scale_val, scale_val),)).to(device)

    pcl = Pointclouds(points=x.unsqueeze(0), features=xf.unsqueeze(0)).to(device)

    img = renderer(pcl, cameras=cam)[0]

    return img