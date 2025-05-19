#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from utils.graphics_utils import getWorld2View2, getProjectionMatrix

from dataclasses import dataclass
@dataclass
class AffineCameraInfo():
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    is_reference_camera: bool
    reference_altitude: np.array
    image_path: str
    image_name: str
    width: int
    height: int
    centerofscene_ECEF: np.array
    affine_coef: np.array
    affine_inter: np.array
    altitude_bounds: np.array
    min_world: np.array
    max_world: np.array
    sun_affine_coef: np.array
    sun_affine_inter: np.array
    camera_to_sun: np.array

    def __getattribute__(self, name):
        blacklist = ['R','T','FovY','FovX']
        if name in blacklist:
            raise AttributeError(f'Hai provato af accedere a {name}')
        return super().__getattribute__(name)
    
    def convert_to_affine_camera(self, image, gt_alpha_mask, data_device):
        return AffineCamera(
            self,
            image,
            gt_alpha_mask,
            data_device,
            is_reference_camera=self.is_reference_camera
        )

@dataclass
class SunCamera():
    world_view_transform: np.array
    full_proj_transform: np.array
    camera_center: np.array
    image_width: int
    image_height: int
    FoVx: float = 1
    FoVy: float = 1

    def ECEF_to_UVA(self, xyz):
        # xyz: (..., 3)
        # We store the affine matrix as a 4x4 matrix transposed (to be compatible with CUDA code)
        At = self.world_view_transform[:3,:3]
        bt = self.world_view_transform[3, :3]
        uva = xyz @ At + bt
        return uva

class StraightThroughHeaviside(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return (input > -0.6).float()

    @staticmethod
    def backward(ctx, grad_output):
        return nn.functional.sigmoid(grad_output+0.6)*(1-nn.functional.sigmoid(grad_output+0.6))
        # return nn.functional.hardtanh(grad_output)

class ShadowMap(nn.Module):
    def __init__(self):
        super(ShadowMap, self).__init__()
        # self.shadow_act = StraightThroughHeaviside.apply
        # self.shadow_act = nn.Sigmoid()
        # self.shadow_act = nn.Hardsigmoid()
        # self.ambient_light = nn.Parameter(torch.tensor([-1.0,-1.0,-1.0], requires_grad=True))
        # self.shadow_linear_pre = nn.Linear(1, 1, bias=True)
        # with torch.no_grad():
        #     self.shadow_linear_pre.weight = nn.Parameter(self.shadow_linear_pre.weight.data*0.0+1)
        #     self.shadow_linear_pre.bias = nn.Parameter(self.shadow_linear_pre.bias.data*0.0+0.0)

        # self.shadow_linear_post = nn.Linear(1, 3, bias=True)
        # with torch.no_grad():
        #     self.shadow_linear_post.weight = nn.Parameter(self.shadow_linear_post.weight.data*0.0+0.5)
        #     self.shadow_linear_post.bias = nn.Parameter(self.shadow_linear_post.bias.data*0.0+0.5)

        # self.densitymap = nn.Parameter(torch.full(shape, 0.4, requires_grad=True))
    
    def forward(self, shadow):
        # shadow = torch.exp(shadow*0.1)
        # shadow = self.shadow_act(shadow)
        # shadow = self.shadow_linear_pre(shadow)
        # shadow = self.shadow_act(shadow+2.9)
        shadow = torch.exp(0.4*shadow.clip(max=0.0))
        # shadow = shadow + (1-shadow)*nn.functional.sigmoid(self.ambient_light)
        # shadow = self.shadow_linear_post(shadow)
        # shadow = shadow.clip(0.01, 1.0)
        return shadow
    
class AffineCamera(nn.Module):
    def __init__(
            self,
            caminfo : AffineCameraInfo,
            image,
            gt_alpha_mask,
            data_device = "cuda",
            is_reference_camera = False
        ):
        super(AffineCamera, self).__init__()

        self.image_name = caminfo.image_name
        self.is_reference_camera = is_reference_camera

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.reference_altitude = torch.from_numpy(caminfo.reference_altitude).to(self.data_device).float()
        self.min_world = torch.from_numpy(caminfo.min_world).to(self.data_device).float()
        self.max_world = torch.from_numpy(caminfo.max_world).to(self.data_device).float()
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        # self.zfar = 100.0
        # self.znear = 0.01

        self.UV_grid = torch.meshgrid(
            torch.linspace(-1,1,self.image_width, device=self.data_device),
            torch.linspace(-1,1,self.image_height, device=self.data_device),
            indexing='xy',
        )

        self.centerofscene_ECEF = torch.from_numpy(caminfo.centerofscene_ECEF).to(self.data_device).float()
        affine_coef = torch.from_numpy(caminfo.affine_coef)
        affine_inter = torch.from_numpy(caminfo.affine_inter)

        # self.affine = torch.eye(4,4)
        # self.affine[:2,:3] = affine_coef[:2]
        # self.affine[:2, 2] = affine_inter[:2]

        self.affine = torch.eye(4,4)
        self.affine[:3,:3] = affine_coef
        self.affine[:3,-1] = affine_inter

        self.affine = self.affine.to(self.data_device).float().T

        self.Ainv = torch.inverse(self.affine[:3,:3].T)


        self.sun_affine = torch.eye(4,4)
        self.sun_affine[:3,:3] = torch.from_numpy(caminfo.sun_affine_coef)
        self.sun_affine[:3,-1] = torch.from_numpy(caminfo.sun_affine_inter)

        self.sun_affine = self.sun_affine.to(self.data_device).float().T

        self._camera_to_sun = torch.from_numpy(caminfo.camera_to_sun).to(self.data_device).float()

        self.altitude_bounds = torch.from_numpy(caminfo.altitude_bounds).to(self.data_device).float()


        self.FoVx = 1
        self.FoVy = 1
        self.world_view_transform = self.affine
        self.full_proj_transform = self.affine
        self.camera_center = torch.zeros(3, device=self.data_device)

        # self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        # self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        # self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        # self.camera_center = self.world_view_transform.inverse()[3, :3]

        # Optimize per-camera color correction
        self.color_correction = nn.Conv2d(3, 3, 1, bias=True).to(self.data_device)
        with torch.no_grad():
            self.color_correction.weight = nn.Parameter(torch.eye(3, device=self.data_device).reshape(3,3,1,1))
            self.color_correction.bias = nn.Parameter(torch.zeros(3, device=self.data_device))

        self.inshadow_color_correction = nn.Parameter(torch.zeros(3, device=self.data_device).reshape(3,1,1)+0.05)

        # # self.inshadow_color_correction = nn.Conv2d(3, 3, 1, bias=False).to(self.data_device)
        # with torch.no_grad():
        #     # self.inshadow_color_correction.weight = nn.Parameter(torch.eye(3, device=self.data_device).reshape(3,3,1,1) * 0.05)
        #     # self.inshadow_color_correction.bias = nn.Parameter(torch.zeros(3, device=self.data_device)+0.2)
        # # self.inshadow_color_correction = nn.Parameter(torch.tensor([0.66], device=self.data_device))

        # Shadow color correction
        self.shadow_map = ShadowMap().to(self.data_device)

        # # Transient mask and canvas
        # self.transient_mask = torch.zeros((self.image_height, self.image_width), device=self.data_device) + 0.01
        # self.transient_mask._requires_grad = True
        # self.transient_mask = nn.Parameter(self.transient_mask)
        # # self.transient_canvas = self.original_image.clone()
        # # self.transient_canvas._requires_grad = True
        # # self.transient_canvas = nn.Parameter(self.transient_canvas)

    def render_pipeline(self, raw_render, sun_altitude_diff=None):
        # raw_render : (3,H,W)
        # sun_altitude_diff : (H,W) or None

        render = raw_render.unsqueeze(0)

        # Step 2: Apply color correction
        if True: #not self.is_reference_camera:
            cc = self.color_correction(render)
        else:
            cc = render

        # Step 1: Apply shades
        if sun_altitude_diff is not None:
            shadow = self.shadow_map(sun_altitude_diff) # Now is between 0 and 1, dimensions are (H,W)
            shaded = shadow * cc + (1-shadow) * self.inshadow_color_correction*cc
        else:
            shadow = None
            shaded = cc

        # # # Step 3: Paint transient on top
        # mask = self.transient_mask.clip(0,1)
        # # final = shaded * (1-mask) + self.transient_canvas * mask
        # final = shaded * (1-mask) + self.original_image * mask
        final = shaded

        return {
            'shadowmap': shadow,
            'shaded': shaded.squeeze(0),
            'cc': cc.squeeze(0),
            'final': final.squeeze(0)
        }

    def get_sun_camera(self, f=2):
        # The sun camera rendering is done with larger footprint (and resolution)
        scalingmat = torch.eye(4, device="cuda")
        scalingmat[0, 0] = 1/f
        scalingmat[1, 1] = 1/f

        cam2virt = scalingmat[:3,:3] @ self._camera_to_sun

        return SunCamera(
            world_view_transform = self.sun_affine @ scalingmat,
            full_proj_transform = self.sun_affine @ scalingmat,
            camera_center = self.camera_center,
            image_width = self.image_width * f,
            image_height = self.image_height * f,
            FoVx = self.FoVx,
            FoVy = self.FoVy
        ), cam2virt

    def get_nadir_camera(self, f=1):
        A = self.affine[:3,:3].T
        b = self.affine[3, :3]

        d = torch.zeros(3, device="cuda")
        d[-1] = 1
        q = A @ d
        q = q/q[-1]
        myM = torch.eye(3, device=self.data_device)
        myM[:2,2] = -q[:2]
        
        new_A = myM @ A
        new_b = (torch.eye(3, device=myM.device)-myM) @ A @ self.centerofscene_ECEF + b

        new_affine = torch.eye(4,4)
        new_affine[:3,:3] = new_A
        new_affine[:3,-1] = new_b
        new_affine = new_affine.to(self.data_device).float().T

        return SunCamera(
            world_view_transform = new_affine,
            full_proj_transform = new_affine,
            camera_center = self.camera_center,
            image_width = self.image_width,
            image_height = self.image_height,
            FoVx = self.FoVx,
            FoVy = self.FoVy
        ), myM

    def sample_random_camera(self, virtual_camera_extent):
        A = self.affine[:3,:3].T
        b = self.affine[3, :3]
        myM = torch.eye(3, device=self.data_device)
        myM[:2,2] = myM[:2,2] + torch.randn(2, device=self.data_device).clip(-1,1)*virtual_camera_extent
        new_A = myM @ A
        new_b = (torch.eye(3, device=myM.device)-myM) @ A @ self.centerofscene_ECEF + b

        new_affine = torch.eye(4,4)
        new_affine[:3,:3] = new_A
        new_affine[:3,-1] = new_b
        new_affine = new_affine.to(self.data_device).float().T

        return SunCamera(
            world_view_transform = new_affine,
            full_proj_transform = new_affine,
            camera_center = self.camera_center,
            image_width = self.image_width,
            image_height = self.image_height,
            FoVx = self.FoVx,
            FoVy = self.FoVy
        ), myM

    def ECEF_to_UVA(self, xyz):
        # xyz: (..., 3)
        # We store the affine matrix as a 4x4 matrix transposed (to be compatible with CUDA code)
        At = self.affine[:3,:3]
        bt = self.affine[3, :3]
        uva = xyz @ At + bt
        return uva

    def UVA_to_ECEF(self, uva):
        # uva: (..., 3)
        # We store the affine matrix as a 4x4 matrix transposed (to be compatible with CUDA code)
        b = self.affine[3, :3].double()
        xyz = torch.einsum('...ij,...j->...i', self.Ainv.double(), uva.double()-b.double())
        return xyz

class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            self.original_image *= gt_alpha_mask.to(self.data_device)
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]

