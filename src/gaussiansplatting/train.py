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

import os
import torch
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from utils.sh_utils import SH2RGB, RGB2SH
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
import iio
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
from scene.cameras import AffineCamera

def render_resample_virtual_camera(virtual_camera, cam2virt, rendered_uva, gaussians, pipe, background, return_extra=False):
    '''
    INPUTS:
        - virtual_camera: the camera from which we render the scene
        - cam2virt: the transformation matrix from the actual camera to the virtual camera (UVA->UVA)
        - rendered_uva: the UVA meshgrid (H,W,3) of the rendered image from the actual camera
        - gaussians, pipe, background: the usual suspects
    OUTPUTS:
        - virtual_rgb_sample: the RGB image of the virtual camera resampled to the actual camera
        - virtual_altitude_sample: the altitude image of the virtual camera resampled to the actual camera
        - virtual_uv: the UVA coordinates where the virtual image was sampled
    '''
    # Step 1. Render the scene from the virtual camera pov
    # the virtual_render_pkg is now a (H,W,3+1) tensor with channels RGB+altitude
    virtual_render_pkg = render(virtual_camera, gaussians, pipe, background)["render"]

    # Step 2. Use the coordinate transformation to reproject the actual image to the virtual image
    virtual_uv = torch.einsum('...ij,...j->...i', cam2virt, rendered_uva)[...,:2]

    # Step 3. Sample the virtual image at the reprojected coordinates
    virtual_render_pkg_sample = torch.nn.functional.grid_sample(
        virtual_render_pkg.unsqueeze(0),
        virtual_uv.unsqueeze(0),
        align_corners=True,
    ).squeeze(0) # (H,W,3+1)

    virtual_rgb_sample = virtual_render_pkg_sample[:3]
    virtual_altitude_sample = virtual_render_pkg_sample[3]

    virtual_altitude_sample[(virtual_uv.abs() > 1).any(-1)] = -100
    if return_extra:
        return virtual_rgb_sample, virtual_altitude_sample, virtual_uv, virtual_render_pkg
    return virtual_rgb_sample, virtual_altitude_sample, virtual_uv

@torch.no_grad()
def render_all_views(cameras, gaussians, pipe, bg=None, override_color=None):
    if bg is None:
        bg = torch.rand((5), device="cuda")

    out = []

    for viewpoint_cam in cameras:
        bg[3] = viewpoint_cam.altitude_bounds[0].item()
        bg[4] = 0.0
        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, override_color=override_color)
        raw_render = render_pkg["render"][:3]
        altitude_render = render_pkg["render"][3]
        rendered_uva = torch.stack(viewpoint_cam.UV_grid+(altitude_render,), dim=-1)

        sun_camera, camera_to_sun = viewpoint_cam.get_sun_camera()
        _, sun_altitude_sample, _ = render_resample_virtual_camera(
            virtual_camera=sun_camera,
            cam2virt=camera_to_sun,
            rendered_uva=rendered_uva,
            gaussians=gaussians,
            pipe=pipe,
            background=bg,
        )
        sun_altitude_diff = altitude_render - sun_altitude_sample

        output = viewpoint_cam.render_pipeline(
            raw_render = raw_render,
            sun_altitude_diff = sun_altitude_diff,
        )

        out.append({
            'image_name': viewpoint_cam.image_name,
            'shadow': output["shadowmap"],
            'raw_render': raw_render,
            'cc': output["cc"],
            'render': output["final"],
            'projxyz': viewpoint_cam.ECEF_to_UVA(gaussians.get_xyz)[:,:2],
            'altitude_render': altitude_render,
        })

    return out

def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1, 1] if dataset.white_background else [0, 0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    ema_Lphotometric_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress", ncols=80)
    first_iter += 1

    color_correction_optimizer = torch.optim.Adam(
        [{'params': s.parameters()} for s in scene.getTrainCameras()],
        lr=1e-2
    )

    init_number_of_gaussians = len(gaussians._xyz)

    for iteration in range(first_iter, opt.iterations + 1):        

        iter_start.record()

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
        viewpoint_cam : AffineCamera = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((5), device="cuda") if opt.random_background else background
        bg[3] = viewpoint_cam.altitude_bounds[0].item()
        bg[4] = 0.0

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg)
        raw_render = render_pkg["render"][:3]
        altitude_render = render_pkg["render"][3]
        accumulated_opacity_render = render_pkg["render"][4]

        rendered_uva = torch.stack(viewpoint_cam.UV_grid+(altitude_render,), dim=-1)

        # Initialize all extra losses to zero
        L_opacity = 0
        L_sun_altitude_resample = 0
        L_sun_rgb_resample = 0
        L_new_altitude_resample = 0
        L_new_rgb_resample = 0
        L_TV_altitude = 0
        L_erank = 0
        L_nll = 0
        L_translucentshadows = 0
        sun_altitude_diff = None
        L_accumulated_opacity = 0

        if iteration > opt.iterstart_L_accumulated_opacity:
            L_accumulated_opacity = (1.0 - accumulated_opacity_render).mean()

        # Sun camera rendering and losses
        if iteration > opt.iterstart_shadowmapping:
            sun_camera, camera_to_sun = viewpoint_cam.get_sun_camera()
            sun_rgb_sample, sun_altitude_sample, sun_uv = render_resample_virtual_camera(
                virtual_camera=sun_camera,
                cam2virt=camera_to_sun,
                rendered_uva=rendered_uva,
                gaussians=gaussians,
                pipe=pipe,
                background=bg,
            )
            sun_altitude_diff = altitude_render - sun_altitude_sample
            if iteration > opt.iterstart_L_sun_resample:
                sun_rgb_diff_map = raw_render - sun_rgb_sample
                sun_visibility_map = (sun_altitude_diff > -1e-2) * (sun_uv.abs() < 1).all(-1)
                sun_visibility_map = sun_visibility_map.detach()
                if sun_visibility_map.any():
                    L_sun_altitude_resample = (sun_altitude_diff.abs() * sun_visibility_map).sum() / (sun_visibility_map.sum())
                    L_sun_rgb_resample = (sun_rgb_diff_map.abs() * sun_visibility_map).sum() / (sun_visibility_map.sum())

        # Random camera rendering and losses
        if iteration > opt.iterstart_L_new_resample:
            new_camera, camera_to_new = viewpoint_cam.sample_random_camera(opt.virtual_camera_extent)
            new_rgb_sample, new_altitude_sample, new_uv = render_resample_virtual_camera(
                virtual_camera=new_camera,
                cam2virt=camera_to_new,
                rendered_uva=rendered_uva,
                gaussians=gaussians,
                pipe=pipe,
                background=bg,
            )
            new_altitude_diff = altitude_render - new_altitude_sample
            new_rgb_diff_map = raw_render - new_rgb_sample
            new_occlusion_map = (new_altitude_diff.abs() < 0.30) * (new_uv.abs() < 1).all(-1)
            new_occlusion_map = new_occlusion_map.detach()
            if new_occlusion_map.any():
                L_new_altitude_resample = (new_altitude_diff.abs() * new_occlusion_map).sum() / (new_occlusion_map.sum())
                L_new_rgb_resample = (new_rgb_diff_map.abs() * new_occlusion_map).sum() / (new_occlusion_map.sum())

        # TV computation
        if iteration > opt.iterstart_L_TV_altitude:
            diff1 = altitude_render[..., 1:, :] - altitude_render[..., :-1, :]
            diff2 = altitude_render[..., :, 1:] - altitude_render[..., :, :-1]
            L_TV_altitude = 0.5*(diff1.abs().mean() + diff2.abs().mean())

        # Opacity loss
        if iteration > opt.iterstart_L_opacity:
            opacity = gaussians.get_opacity.squeeze()
            L_opacity = opacity.sum() / init_number_of_gaussians

        if iteration > opt.iterstart_L_erank:
            s2 = gaussians.get_scaling.square()+1e-5
            S = s2.sum(dim=1, keepdim=True)
            q = s2 / S
            erankm1 = torch.expm1(-(q * torch.log(q + 1e-6)).sum(dim=1))
            L_erank = (torch.log(erankm1+1e-5).mul(-1).clip(min=0.0)+s2.amin(1).sqrt()).mean()

        output = viewpoint_cam.render_pipeline(
            raw_render = raw_render,
            sun_altitude_diff = sun_altitude_diff,
        )
        # image = output["final"]
        image = output["shaded"]

        if output['shadowmap'] is not None:
            shadowmap = output['shadowmap']
            a = shadowmap
            b = shadowmap.clip(0.05, 0.95)
            L_translucentshadows = -(a * torch.log2(b) + (1-a) * torch.log2(1-b)).mean()
            # L_translucentshadows = (shadowmap*(1-shadowmap)).mean()

        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        Ll1 = l1_loss(image, gt_image)
        # Ll1rel = (image - gt_image).abs() / (gt_image.abs() + 1/255)
        # Ll1rel = Ll1rel.mean()
        Lphotometric = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        L_altitude_reference = (altitude_render - viewpoint_cam.reference_altitude).abs().mean()
        if iteration > args.iterstart_L_nll:
            betaprime = (viewpoint_cam.transient_mask.clip(0,1)+0.05).square()
            L_nll = torch.nn.functional.gaussian_nll_loss(
                input= image,
                target= gt_image,
                var = betaprime.unsqueeze(0).repeat(3,1,1),
            )
        loss = (
            Lphotometric
            # + L_altitude_reference
            + opt.w_L_opacity * L_opacity
            + opt.w_L_sun_altitude_resample * L_sun_altitude_resample
            + opt.w_L_sun_rgb_resample * L_sun_rgb_resample
            + opt.w_L_new_altitude_resample * L_new_altitude_resample
            + opt.w_L_new_rgb_resample * L_new_rgb_resample
            + opt.w_L_TV_altitude * L_TV_altitude
            + opt.w_L_erank * L_erank
            + opt.w_L_nll * L_nll
            + opt.w_L_translucentshadows * L_translucentshadows
            + opt.w_L_accumulated_opacity * L_accumulated_opacity
        )
        # # Add a high constrast loss
        # inverse_constrast = 0.045
        # inverse_brightness = 0.045
        # gt_image_high_contrast = -inverse_constrast + gt_image.clip(0,1) * (2*inverse_constrast+inverse_brightness)
        # image_high_contrast = -inverse_constrast + image.clip(0,1) * (2*inverse_constrast+inverse_brightness)
        # loss = loss + l1_loss(image_high_contrast, gt_image_high_contrast)
        loss = loss.mean()
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            ema_Lphotometric_for_log = 0.4 * Lphotometric.item() + 0.6 * ema_Lphotometric_for_log
            if iteration % 10 == 0:
                progress_bar.set_postfix({
                    "Loss": f"{ema_loss_for_log:.{7}f}",
                    "Lphotometric": f"{ema_Lphotometric_for_log:.{7}f}",
                })
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Logging
            if tb_writer:
                tb_writer.add_scalar('number of gaussians', len(gaussians._xyz), iteration)
                tb_writer.add_scalar('Lphotometric', Lphotometric, iteration)
                tb_writer.add_scalar('LNLL', L_nll, iteration)
                tb_writer.add_scalar('Ll1', Ll1, iteration)
                tb_writer.add_scalar('TotalLoss', loss, iteration)
                tb_writer.add_scalar('Lopacity', L_opacity, iteration) 
                tb_writer.add_scalar('meanopacity', opacity.mean().item(), iteration) 
                tb_writer.add_scalar('L_sun_altitude_resample', L_sun_altitude_resample, iteration) 
                tb_writer.add_scalar('L_sun_rgb_resample', L_sun_rgb_resample, iteration) 
                tb_writer.add_scalar('L_new_altitude_resample', L_new_altitude_resample, iteration) 
                tb_writer.add_scalar('L_new_rgb_resample', L_new_rgb_resample, iteration) 
                tb_writer.add_scalar('L_TV_altitude', L_TV_altitude, iteration) 
                tb_writer.add_scalar('L_altitude_reference', L_altitude_reference, iteration)
                tb_writer.add_scalar('L_erank', L_erank, iteration)

            # Color normalization before saving for the last time
            if iteration == opt.iterations:
                view = None
                for c in scene.getTrainCameras():
                    if c.is_reference_camera:
                        view = c
                        break 
                assert view is not None

                A1 = view.color_correction.weight.squeeze()
                b1 = view.color_correction.bias.squeeze()
                A1inv = torch.linalg.inv(A1.double()).float()

                rgb_colors = SH2RGB(gaussians._features_dc)
                normalized_rgb_colors = torch.einsum('ij,...j->...i', A1, rgb_colors) + b1
                gaussians._features_dc = RGB2SH(normalized_rgb_colors)

                for c in scene.getTrainCameras():
                    Ai = c.color_correction.weight.squeeze()
                    bi = c.color_correction.bias.squeeze()
                    AiA1inv = Ai @ A1inv
                    AiAiinvb1 = Ai @ A1inv @ b1
                    c.color_correction.weight.data = AiA1inv.reshape(3,3,1,1)
                    c.color_correction.bias.data = -AiAiinvb1 + bi

                print('done global cc alignment')

            # Saving
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)
                # Now save also the color correction
                color_correction_path = os.path.join(scene.model_path, "color_correction/iteration_{}".format(iteration))
                os.makedirs(color_correction_path, exist_ok=True)
                torch.save(
                    [{
                        'image_name': c.image_name,
                        'state_dict': c.state_dict(),
                    }
                    for c in scene.getTrainCameras()],
                    os.path.join(color_correction_path, "color_correction.pth")
                )
                # Save the optimizer state
                optimizer_path = os.path.join(scene.model_path, "optimizer/iteration_{}".format(iteration))
                os.makedirs(optimizer_path, exist_ok=True)
                torch.save(
                    {
                        'gaussians': gaussians.optimizer.state_dict(),
                        'color_correction': color_correction_optimizer.state_dict(),
                    },
                    os.path.join(optimizer_path, "optimizer.pth")
                )

            # Optimizer step
            if iteration < opt.iterations:
                # Recale the gradients of xyz with respect to the altitude
                # grad = gaussians.get_xyz.grad
                # grad_uva = grad @ viewpoint_cam.affine[:3,:3]
                # grad_uva[...,2] = grad_uva[...,2] * 30.0
                # grad = grad_uva @ torch.linalg.inv(viewpoint_cam.affine[:3,:3])
                gaussians.optimizer.step()
                color_correction_optimizer.step()
                gaussians.update_learning_rate(iteration)
                # if iteration % 5 == 0 and iteration < 1000:
                #     grad_norm = gaussians._opacity.grad.clone().detach().norm(dim=1)
                #     z = gaussians._xyz[...,2].clone().detach()
                #     os.makedirs(os.path.join(scene.model_path, "grad_norm"), exist_ok=True)
                #     torch.save(
                #         (z.detach().cpu(), grad_norm.detach().cpu()),
                #         os.path.join(scene.model_path, "grad_norm", f"grad_norm_{iteration:05}.pth")
                #     )
                gaussians.optimizer.zero_grad(set_to_none = True)
                color_correction_optimizer.zero_grad(set_to_none = True)
            
            # Pruining
            transparent_mask = gaussians._opacity.squeeze() < -6.0
            if transparent_mask.any():
                gaussians.prune_points(transparent_mask)

            if iteration == opt.color_reset_iterations:
                to_reset = torch.full((gaussians._xyz.size(0),), False, device="cuda", dtype=bool)
                myoutput = render_all_views(scene.getTrainCameras(), gaussians, pipe)
                for myoutputview in myoutput:
                    shadowmap = myoutputview["shadow"]
                    pts = myoutputview["projxyz"]
                    to_reset_iter = 1-torch.max_pool2d(1-shadowmap[None,None], 5, stride=1, padding=2).squeeze()
                    to_reset_iter = torch.nn.functional.grid_sample(
                        to_reset_iter[None,None],
                        pts[None,None],
                        mode='bilinear',
                        align_corners=True,
                        padding_mode='zeros',
                    ).squeeze() < 0.5
                    to_reset = to_reset | to_reset_iter

                with torch.no_grad():
                    # gaussians._opacity[to_reset] = gaussians.opacity_activation(gaussians._opacity[to_reset])
                    # gaussians._opacity[to_reset] = gaussians._opacity[to_reset]*0.05 + 0.01
                    # gaussians._opacity[to_reset] = gaussians.inverse_opacity_activation(gaussians._opacity[to_reset])
                    
                    gaussians._opacity[to_reset] = gaussians.inverse_opacity_activation(0.005 * torch.ones_like(gaussians._opacity[to_reset]))
                    gaussians._features_dc[to_reset] = RGB2SH(torch.full_like(gaussians._features_dc[to_reset], 1.1))
                    gaussians._scaling[to_reset] = gaussians.scaling_inverse_activation((1.0/400) * torch.ones_like(gaussians._scaling[to_reset])) # TODO: the scaling should be reset using kNN
                    # gaussians._rotation[to_reset] = torch.tensor([1.0,0.0,0.0,0.0], device="cuda").repeat(to_reset.sum(),1)
                    # gaussians._xyz[to_reset] = torch.randn_like(gaussians._xyz[to_reset]) * 2.0 + gaussians._xyz[to_reset]

                    for param in [
                        gaussians._opacity,
                        gaussians._features_dc,
                        gaussians._scaling,
                        # gaussians._rotation,
                        # gaussians._xyz,
                    ]:
                        mask = to_reset.squeeze().clone()
                        while len(mask.shape) < len(param.shape):
                            mask = mask.unsqueeze(-1)
                        gaussians.optimizer.state[param]['exp_avg'].masked_fill_(mask, 0.0)
                        gaussians.optimizer.state[param]['exp_avg_sq'].masked_fill_(mask, 0.0)

            if iteration in testing_iterations:
                test_background = torch.tensor([1.0, 0.0, 1.0, -100.0], dtype=torch.float32, device="cuda")
                # test_background[3] = viewpoint_cam.altitude_bounds[0].item()
                nardir_cam = [c for c in scene.getTestCameras() if 'Nadir' in c.image_name]
                out = render_all_views(nardir_cam, gaussians, pipe, bg=test_background)
                altitude_render = out[0]["altitude_render"] # (H,W)
                altitude_render = torch.where(
                    altitude_render < viewpoint_cam.altitude_bounds[0].item(),
                    torch.nan,
                    altitude_render,
                )
                altitude_render = torch.where(
                    altitude_render > viewpoint_cam.altitude_bounds[1].item(),
                    torch.nan,
                    altitude_render,
                )
                os.makedirs(os.path.join(scene.model_path, "altitude_records"), exist_ok=True)
                iio.write(
                    os.path.join(scene.model_path, "altitude_records", f"altitude_render_{iteration:05}.iio"),
                    altitude_render.detach().cpu().numpy(),
                )

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                l1_test /= len(config['cameras'])          
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    test_iterations_default = list(range(0,100))+list(range(100, 1000, 10))+list(range(1000, 10000, 50))
    test_iterations_default = sorted(list(set(test_iterations_default)))
    test_iterations_default = []
    parser.add_argument("--test_iterations", nargs="+", type=int, default=test_iterations_default)
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
