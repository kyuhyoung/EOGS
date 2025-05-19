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
from scene import Scene
import os
from gaussian_renderer import render
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from train import render_resample_virtual_camera

from scene.cameras import SunCamera

def render_video(dataset : ModelParams, iteration : int, pipeline : PipelineParams, opacity_treshold : float = None):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
        iteration = scene.loaded_iter

        print('gaussians._xyz.shape', gaussians._xyz.shape)

        if opacity_treshold is not None:
            assert abs(opacity_treshold) <= 1
            if opacity_treshold < 0:
                # Tengo le gaussiane con opacità piccola
                invalid = gaussians.get_opacity > -opacity_treshold
            else:
                # Tengo le gaussiane con opacità grande
                invalid = gaussians.get_opacity < opacity_treshold
            gaussians._opacity[invalid] = -20.0
        # gaussians._opacity.fill_(gaussians.inverse_opacity_activation(torch.tensor(0.1, device=gaussians._xyz.device)))

        altitude_min, altitude_max = scene.getTrainCameras()[0].altitude_bounds.cpu().detach().numpy().tolist()
        bg_color = [1,0,1,altitude_min,0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        ###############
        # Temporary fix for the fact that the colors are not yet normalized
        ###############
        cc = torch.load(os.path.join(dataset.model_path, "color_correction", f"iteration_{iteration}", "color_correction.pth"), 'cpu')
        cc_cam = scene.getTrainCameras()[0]
        found = False
        for j in range(len(cc)):
            if cc_cam.image_name == cc[j]["image_name"]:
                cc_cam.load_state_dict(cc[j]["state_dict"])
                cc_cam = cc_cam.to(gaussians._xyz.device)
                found = True
                break
        assert found
        ###############

        # From cc_cam take the sun position
        sun_camera, cc2sun = cc_cam.get_sun_camera()

        # nadir_cam = scene.getTestCameras()[-1]
        # assert nadir_cam.image_name == 'Nadir'

        # Now we samples affine camera in a circle around the nadir camera
        azimut_angles = torch.linspace(0, 2 * torch.pi, 200)
        altitude_angle = torch.deg2rad(torch.tensor(45.0))
        images = []
        for azimut in azimut_angles:
            # Compute the vector direction corresponding to the azimut and altitude angles
            x = torch.cos(azimut) * torch.cos(altitude_angle)
            y = torch.sin(azimut) * torch.cos(altitude_angle)
            z = torch.sin(altitude_angle)
            d = torch.stack([x, y, z], dim=0).to(cc_cam.data_device)
            d = d / d.norm()

            A = cc_cam.affine[:3,:3].T
            b = cc_cam.affine[3, :3]

            q = A @ d
            q = q/q[-1]
            myM = torch.eye(3, device=cc_cam.data_device)
            myM[:2,2] = -q[:2]
            
            new_A = myM @ A
            new_b = (torch.eye(3, device=myM.device)-myM) @ A @ cc_cam.centerofscene_ECEF + b

            view2cc = torch.linalg.inv(myM.double()).float()

            new_affine = torch.eye(4,4)
            new_affine[:3,:3] = new_A
            new_affine[:3,-1] = new_b
            new_affine = new_affine.to(cc_cam.data_device).float().T

            view = SunCamera(
                world_view_transform = new_affine,
                full_proj_transform = new_affine,
                camera_center = cc_cam.camera_center,
                image_width = cc_cam.image_width,
                image_height = cc_cam.image_height,
                FoVx = cc_cam.FoVx,
                FoVy = cc_cam.FoVy
            )

            render_pkg = render(view, gaussians, pipeline, background)["render"]
            raw_render = render_pkg[:3]
            altitude_render = render_pkg[3]

            rendered_uva = torch.stack(cc_cam.UV_grid+(altitude_render,), dim=-1)


            cam2virt=cc2sun @ view2cc

            _, sun_altitude_sample, _, _ = render_resample_virtual_camera(
                virtual_camera=sun_camera,
                cam2virt=cam2virt,
                rendered_uva=rendered_uva,
                gaussians=gaussians,
                pipe=pipeline,
                background=background,
                return_extra=True,
            )
            sun_altitude_diff = altitude_render - sun_altitude_sample

            renderings = cc_cam.render_pipeline(
                raw_render = raw_render,
                sun_altitude_diff = sun_altitude_diff,
            )

            final = renderings["final"].detach().cpu() # 3xHxW
            elevation = altitude_render.detach().cpu().squeeze().repeat(3,1,1) # 3xHxW
            elevation = (elevation - altitude_min) / (altitude_max - altitude_min)

            image = torch.cat([final, elevation], dim=-1)

            images.append(image)
        
        print(len(images))
        import iio
        print(image.shape)
        iio.write('cose.iio', image.squeeze().permute(1,2,0).numpy())
        # Create a video given the N images (each has shape 3xHxW)
        import cv2
        H, W = images[0].shape[1:]
        out = cv2.VideoWriter(
            filename='nadir_pov.avi',
            fourcc=cv2.VideoWriter_fourcc(*'DIVX'),
            fps=30, 
            frameSize=(W, H),
            isColor=True,
        )
        for i in range(len(images)):
            img = images[i].permute(1,2,0).clip(0,1).mul(255.0).numpy()
            img = cv2.cvtColor(img.astype('uint8'), cv2.COLOR_RGB2BGR)
            out.write(img)
        out.release()



if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--opacity_treshold", default='')
    parser.add_argument("--res", default=0.5, type=float)
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    if args.opacity_treshold != '':
        opacity_treshold = float(args.opacity_treshold)
    else:
        opacity_treshold = None

    render_video(model.extract(args), args.iteration, pipeline.extract(args), opacity_treshold=opacity_treshold)
