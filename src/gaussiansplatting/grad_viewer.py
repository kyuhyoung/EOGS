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
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import iio
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.loss_utils import l1_loss
from train import render_resample_virtual_camera

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):
    # Prepare the output directories
    base_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    # for kind in ["renders", "altitude", "sunpov", "sunpovaltitude", "shadowmap", "shaded", "cc", "sunaltitudesampled", "gt"]:
    for kind in ["xyz","scaling"]:
        makedirs(os.path.join(base_path, "gradient", kind), exist_ok=True)

    # Load the color corrections
    cc = torch.load(os.path.join(model_path, "color_correction", f"iteration_{iteration}", "color_correction.pth"), 'cpu')
    for i in range(len(views)):
        for j in range(len(cc)):
            if views[i].image_name == cc[j]["image_name"]:
                views[i].load_state_dict(cc[j]["state_dict"])
                views[i] = views[i].to(gaussians._xyz.device) 

    original_colors = gaussians._features_dc.clone()                   

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        # Reset gradients
        gaussians._xyz.grad = None
        gaussians._features_dc.grad = None
        gaussians._features_rest.grad = None
        gaussians._scaling.grad = None
        gaussians._rotation.grad = None
        gaussians._opacity.grad = None

        # Reset colors
        gaussians._features_dc = original_colors.clone()

        name = "{0:05d}.iio".format(idx)

        background = [1,0,1, view.altitude_bounds[0].item()]
        background = torch.tensor(background, dtype=torch.float32, device="cuda")
        render_pkg = render(view, gaussians, pipeline, background)["render"]
        raw_render = render_pkg[:3]
        altitude_render = render_pkg[3]

        rendered_uva = torch.stack(view.UV_grid+(altitude_render,), dim=-1)

        # Rendering with sun pov
        sun_camera, camera_to_sun = view.get_sun_camera()
        sun_rgb_sample, sun_altitude_sample, _, sunpov = render_resample_virtual_camera(
            virtual_camera=sun_camera,
            cam2virt=camera_to_sun,
            rendered_uva=rendered_uva,
            gaussians=gaussians,
            pipe=pipeline,
            background=background,
            return_extra=True,
        )
        sun_altitude_diff = altitude_render - sun_altitude_sample

        renderings = view.render_pipeline(
            raw_render = raw_render,
            sun_altitude_diff = sun_altitude_diff,
        )

        gt_image = view.original_image.cuda()
        Ll1 = l1_loss(renderings['final'], gt_image)
        Ll1.backward()

        # Handle xyz gradients
        grad = gaussians._scaling.grad.clone().detach().unsqueeze(1)
        grad = torch.cat([grad,torch.ones_like(grad[..., :1])], dim=-1)
        print(grad.abs().max())

        # Reset gradients
        gaussians._xyz.grad = None
        gaussians._features_dc.grad = None
        gaussians._features_rest.grad = None
        gaussians._scaling.grad = None
        gaussians._rotation.grad = None
        gaussians._opacity.grad = None

        background = [0,0,0,0]
        background = torch.tensor(background, dtype=torch.float32, device="cuda")
        grad_raw_render = render(view, gaussians, pipeline, background, override_color=grad)["render"]

        iio.write(os.path.join(base_path, "gradient", "scaling", name), grad_raw_render[:3].detach().permute(1,2,0).cpu().numpy())


        # rendered_uva = torch.stack(view.UV_grid+(altitude_render,), dim=-1)

        # # Rendering with sun pov
        # sun_camera, camera_to_sun = view.get_sun_camera()
        # sun_rgb_sample, sun_altitude_sample, _, sunpov = render_resample_virtual_camera(
        #     virtual_camera=sun_camera,
        #     cam2virt=camera_to_sun,
        #     rendered_uva=rendered_uva,
        #     gaussians=gaussians,
        #     pipe=pipeline,
        #     background=background,
        #     return_extra=True,
        # )
        # sun_altitude_diff = altitude_render - sun_altitude_sample

        # renderings = view.render_pipeline(
        #     raw_render = raw_render,
        #     sun_altitude_diff = sun_altitude_diff,
        # )

        # # Now save the images
        # iio.write(os.path.join(base_path, "rawrender", name), raw_render.permute(1,2,0).cpu().numpy())
        # iio.write(os.path.join(base_path, "shaded", name), renderings['shaded'].permute(1,2,0).cpu().numpy())
        # iio.write(os.path.join(base_path, "cc", name), renderings['cc'].permute(1,2,0).cpu().numpy())
        # iio.write(os.path.join(base_path, "final", name), renderings['final'].permute(1,2,0).cpu().numpy())
        # iio.write(os.path.join(base_path, "sunpovsampled", name), sun_rgb_sample.permute(1,2,0).cpu().numpy())
        # iio.write(os.path.join(base_path, "gt", name), gt.permute(1,2,0).cpu().numpy())
        # iio.write(os.path.join(base_path, "altitude", name), altitude_render.cpu().numpy())
        # iio.write(os.path.join(base_path, "sunaltitudesampled", name), sun_altitude_sample.cpu().numpy())
        # iio.write(os.path.join(base_path, "sun_altitude_diff", name), sun_altitude_diff.cpu().numpy())
        # iio.write(os.path.join(base_path, "shadowmap", name), renderings['shadow'].cpu().numpy())
        # iio.write(os.path.join(base_path, "sunpov", name), sunpov[:3].permute(1,2,0).cpu().numpy())
        # iio.write(os.path.join(base_path, "sunpovaltitude", name), sunpov[3].cpu().numpy())

        
        # save_image(raw_render, os.path.join(base_path, "rawrender", name))
        # save_image(renderings['shaded'], os.path.join(base_path, "shaded", name))
        # save_image(renderings['cc'], os.path.join(base_path, "cc", name))
        # save_image(renderings['final'], os.path.join(base_path, "final", name))
        # save_image(sun_raw_render, os.path.join(base_path, "sunpov", name))
        # save_image(sun_raw_render_sampled, os.path.join(base_path, "sunpovsampled", name))
        # save_image(gt, os.path.join(base_path, "gt", name))

        # # Now save the numpy arrays
        # np.save(os.path.join(base_path, "altitude", name), altitude_render.detach().cpu().numpy())
        # np.save(os.path.join(base_path, "sunpovaltitude", name), sun_altitude_render.detach().cpu().numpy())
        # np.save(os.path.join(base_path, "sunaltitudesampled", name), sun_altitude_render_sampled.detach().cpu().numpy())
        # np.save(os.path.join(base_path, "shadowmap", name), raw_shadow_map.detach().cpu().numpy())


def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, opacity_treshold : float = None):
    gaussians = GaussianModel(dataset.sh_degree, dataset.scale_bias)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

    print('gaussians._xyz.shape', gaussians._xyz.shape)

    # bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
    # bg_color = [1,0,1,scene.getTrainCameras()[0].altitude_bounds[0].item()]
    bg_color = [0,0,0,scene.getTrainCameras()[0].altitude_bounds[0].item()]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    if not skip_train:
            render_set(dataset.model_path, f"train_op{opacity_treshold}", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

    # if not skip_test:
    #         render_set(dataset.model_path, f"test_op{opacity_treshold}", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

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
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    if args.opacity_treshold != '':
        opacity_treshold = float(args.opacity_treshold)
    else:
        opacity_treshold = None
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, opacity_treshold=opacity_treshold)