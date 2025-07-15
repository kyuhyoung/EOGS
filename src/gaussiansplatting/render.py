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

from train import render_resample_virtual_camera

import rasterio
from plyflatten import plyflatten
from plyflatten.utils import rasterio_crs, crs_proj
import affine
import numpy as np
from PIL import Image

def get_exact_file_name_from_path(str_path):
    return os.path.splitext(os.path.basename(str_path))[0]

def save_image(dir_save, id_img, mat):
    is_rgb = 3 == len(mat.shape)
    path_img = os.path.join(dir_save, f'{id_img}.png')
    v_min = np.nanmin(mat);  v_max = np.nanmax(mat)
    #print(f'mat.shape : {mat.shape}, v_min : {v_min}, v_max : {v_max}')
    if 0 <= v_min and v_max <= 1:
        #print('aa1')
        mat_no_nan = np.where(np.isnan(mat), 0.0, mat)
        im_uint8 = (mat_no_nan * 255).astype(np.uint8)
        #print('aa2')
    else:
        #print('bb1')
        norm = (mat - v_min) / (v_max - v_min) 
        norm_no_nan = np.where(np.isnan(norm), 0.0, norm)
        #print('bb2')
        im_uint8 = (norm_no_nan * 255).clip(0, 255).astype(np.uint8)
    if is_rgb:
        # 1) 채널(axis=2) 중 하나라도 NaN인 픽셀 위치 찾기
        nan_pxl_mask = np.isnan(mat).any(axis = 2)
        # 2) True 갯수 세기
        nan_pxl_count = np.count_nonzero(nan_pxl_mask)
        img = Image.fromarray(im_uint8, mode = 'RGB')
    else:
        nan_pxl_count = np.isnan(mat).sum()
        img = Image.fromarray(im_uint8, mode = 'L')
        
    img.save(path_img)
    #print(f'{path_img} is saved')
    #exit(1)
    n_pxl_total = mat.shape[0] * mat.shape[1]
    ratio_nan_pxl = nan_pxl_count / n_pxl_total
    return nan_pxl_count, ratio_nan_pxl

def render_set(model_path, name, iteration, views, gaussians, pipeline, background, scene_params, resolution):
    #print(f'model_path : {model_path}, name : {name}');    exit(1)
    # Prepare the output directories
    base_path = os.path.join(model_path, name, "ours_{}".format(iteration))
    # for kind in ["renders", "altitude", "sunpov", "sunpovaltitude", "shadowmap", "shaded", "cc", "sunaltitudesampled", "gt"]:
    for kind in [
        "rawrender",
        "shaded",
        "cc",
        "final",
        "gt",
        "altitude",
        "sunaltitudesampled",
        "sunpovsampled",
        "sun_altitude_diff",
        "shadowmap",
        "sunpov",
        "sunpovaltitude",
        "dsm",
        "accumulated_opacity",
        "nadiraltitudesampled",
        "nadirpovsampled",
        "nadir_altitude_diff",
        "nadirpov",
        "nadirpovaltitude",
    ]:
        makedirs(os.path.join(base_path, kind), exist_ok=True)

    # Load the color corrections
    cc = torch.load(os.path.join(model_path, "color_correction", f"iteration_{iteration}", "color_correction.pth"), 'cpu')
    for i in range(len(views)):
        for j in range(len(cc)):
            if views[i].image_name == cc[j]["image_name"]:
                views[i].load_state_dict(cc[j]["state_dict"])
                views[i] = views[i].to(gaussians._xyz.device)                    

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):
        name = "{0:05d}.iio".format(idx)
        print(f'view.image_name : {view.image_name}');  #exit(1)
        id_image = get_exact_file_name_from_path(view.image_name)
        gt = view.original_image[0:3, :, :]

        render_pkg = render(view, gaussians, pipeline, background)["render"]
        raw_render = render_pkg[:3]
        altitude_render = render_pkg[3]
        accumulated_opacity_render = render_pkg[4]

        rendered_uva = torch.stack(view.UV_grid+(altitude_render,), dim=-1)
        # rendered_uva back to utm space
        cloud = view.UVA_to_ECEF(rendered_uva.detach().reshape((-1, 3))).cpu().numpy()

        # Rendering with sun pov
        sun_camera, cam2sun = view.get_sun_camera()
        sun_rgb_sample, sun_altitude_sample, _, sunpov = render_resample_virtual_camera(
            virtual_camera=sun_camera,
            cam2virt=cam2sun,
            rendered_uva=rendered_uva,
            gaussians=gaussians,
            pipe=pipeline,
            background=background,
            return_extra=True,
        )
        sun_altitude_diff = altitude_render - sun_altitude_sample

        # Rendering with nadir pov
        nadir_camera, cam2nadir = view.get_nadir_camera()
        nadir_rgb_sample, nadir_altitude_sample, _, nadirpov = render_resample_virtual_camera(
            virtual_camera=nadir_camera,
            cam2virt=cam2nadir,
            rendered_uva=rendered_uva,
            gaussians=gaussians,
            pipe=pipeline,
            background=background,
            return_extra=True,
        )
        nadir_altitude_diff = altitude_render - nadir_altitude_sample

        renderings = view.render_pipeline(
            raw_render = raw_render,
            sun_altitude_diff = sun_altitude_diff,
        )

        # Now save the images
        iio.write(os.path.join(base_path, "rawrender", name), raw_render.permute(1, 2, 0).cpu().numpy())
        #   (815, 746, 3), 0.00789, 0.75136
        dir_out = os.path.join(base_path, "rawrender")
        save_image(dir_out, f'{id_image}_{idx:02d}', raw_render.permute(1, 2, 0).cpu().numpy())
        iio.write(os.path.join(base_path, "accumulated_opacity", name), accumulated_opacity_render.cpu().numpy())
        #   (815, 746), 0.675, 0.9999 
        dir_out = os.path.join(base_path, "accumulated_opacity")
        save_image(dir_out, f'{id_image}_{idx:02d}', accumulated_opacity_render.cpu().numpy())
        iio.write(os.path.join(base_path, "sunpovsampled", name), sun_rgb_sample.permute(1, 2, 0).cpu().numpy())
        #   (815, 746, 3), 0.0078, 0.804
        dir_out = os.path.join(base_path, "sunpovsampled")
        save_image(dir_out, f'{id_image}_{idx:02d}', sun_rgb_sample.permute(1, 2, 0).cpu().numpy())
        iio.write(os.path.join(base_path, "nadirpovsampled", name), nadir_rgb_sample.permute(1, 2, 0).cpu().numpy())
        #   (815, 746, 3), 0.00, 0.75369
        dir_out = os.path.join(base_path, "nadirpovsampled")
        save_image(dir_out, f'{id_image}_{idx:02d}', nadir_rgb_sample.permute(1, 2, 0).cpu().numpy())
        iio.write(os.path.join(base_path, "gt", name), gt.permute(1, 2, 0).cpu().numpy())
        #   (815, 746, 3), 0.00, 1.0
        dir_out = os.path.join(base_path, "gt")
        save_image(dir_out, f'{id_image}_{idx:02d}', gt.permute(1, 2, 0).cpu().numpy())
        iio.write(os.path.join(base_path, "altitude", name), altitude_render.cpu().numpy())
        #t0.shape : (815, 746), t0.min() : -27.257511138916016, t0.max() : 23.55037498474121
        dir_out = os.path.join(base_path, "altitude")
        save_image(dir_out, f'{id_image}_{idx:02d}', altitude_render.cpu().numpy())
        iio.write(os.path.join(base_path, "sunaltitudesampled", name), sun_altitude_sample.cpu().numpy())
        #   t0.shape : (815, 746), t0.min() : -29.81124496459961, t0.max() : 24.271461486816406
        dir_out = os.path.join(base_path, "sunaltitudesampled")
        save_image(dir_out, f'{id_image}_{idx:02d}', sun_altitude_sample.cpu().numpy())
        iio.write(os.path.join(base_path, "sun_altitude_diff", name), sun_altitude_diff.cpu().numpy())
        #   t0.shape : (815, 746), t0.min() : -41.69911193847656, t0.max() : 26.376760482788086
        dir_out = os.path.join(base_path, "sun_altitude_diff")
        save_image(dir_out, f'{id_image}_{idx:02d}', sun_altitude_diff.cpu().numpy())
        iio.write(os.path.join(base_path, "sunpov", name), sunpov[:3].permute(1, 2, 0).cpu().numpy())
        #t0.shape : (1630, 1492, 3), t0.min() : -0.06879816204309464, t0.max() : 0.8913784027099609
        dir_out = os.path.join(base_path, "sunpov")
        save_image(dir_out, f'{id_image}_{idx:02d}', sunpov[:3].permute(1, 2, 0).cpu().numpy())
        iio.write(os.path.join(base_path, "sunpovaltitude", name), sunpov[3].cpu().numpy())
        #   t0.shape : (1630, 1492), t0.min() : -54.98023223876953, t0.max() : 43.10971450805664
        dir_out = os.path.join(base_path, "sunpovaltitude")
        save_image(dir_out, f'{id_image}_{idx:02d}', sunpov[3].cpu().numpy())
        iio.write(os.path.join(base_path, "nadiraltitudesampled", name), nadir_altitude_sample.cpu().numpy())
        #   t0.shape : (815, 746), t0.min() : -100.0, t0.max() : 23.48965072631836
        dir_out = os.path.join(base_path, "nadiraltitudesampled")
        save_image(dir_out, f'{id_image}_{idx:02d}', nadir_altitude_sample.cpu().numpy())
        iio.write(os.path.join(base_path, "nadir_altitude_diff", name), nadir_altitude_diff.cpu().numpy())
        #   t0.shape : (815, 746), t0.min() : -38.47461700439453, t0.max() : 99.61248779296875
        dir_out = os.path.join(base_path, "nadir_altitude_diff")
        save_image(dir_out, f'{id_image}_{idx:02d}', nadir_altitude_diff.cpu().numpy())
        iio.write(os.path.join(base_path, "nadirpov", name), nadirpov[:3].permute(1, 2, 0).cpu().numpy())
        #   t0.shape : (815, 746, 3), t0.min() : 0.0034999505151063204, t0.max() : 0.7542663216590881
        dir_out = os.path.join(base_path, "nadirpov")
        save_image(dir_out, f'{id_image}_{idx:02d}', nadirpov[:3].permute(1, 2, 0).cpu().numpy())
        iio.write(os.path.join(base_path, "nadirpovaltitude", name), nadirpov[3].cpu().numpy())
        #   t0.shape : (815, 746), t0.min() : -27.448762893676758, t0.max() : 23.676664352416992
        dir_out = os.path.join(base_path, "nadirpovaltitude")
        save_image(dir_out, f'{id_image}_{idx:02d}', nadirpov[3].cpu().numpy())
        for key in renderings:
            makedirs(os.path.join(base_path, key), exist_ok=True)
            out_tmp = renderings[key]
            if len(out_tmp.shape) == 3:
                out_tmp = out_tmp.permute(1, 2, 0)
            iio.write(os.path.join(base_path, key, name), out_tmp.cpu().numpy())
            dir_out = os.path.join(base_path, key)
            save_image(dir_out, f'{id_image}_{idx:02d}', out_tmp.cpu().numpy())
            #   key : shadowmap, t0.shape : (815, 746), t0.min() : 5.7032444544802274e-08, t0.max() : 1.0 [15/07 06:16:46]
            #   key : shaded, t0.shape : (815, 746, 3), t0.min() : 0.0023985973093658686, t0.max() : 0.751362681388855 [15/07 06:16:46]
            #   key : cc, t0.shape : (815, 746, 3), t0.min() : 0.007893809117376804, t0.max() : 0.751362681388855 [15/07 06:16:46]
            #   key : final, t0.shape : (815, 746, 3), t0.min() : 0.0023985973093658686, t0.max() : 0.751362681388855 [15/07 06:16:46]
        #exit(1);
        
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

        # Unnormalized the point cloud so we're in normal utm again
        cloud = cloud * scene_params[1] + scene_params[0]

        import numpy as np
        # TODO: Each dataset has its own meter/pixel resolution
        if 'IARPA' in base_path:
            resolution=0.3
        elif 'JAX' in base_path:
            resolution=0.5
        else:
            raise ValueError('Unknown dataset')
        xmin, xmax = cloud[:, 0].min(), cloud[:, 0].max()
        ymin, ymax = cloud[:, 1].min(), cloud[:, 1].max()
        xoff = np.floor(xmin / resolution) * resolution
        xsize = int(1 + np.floor((xmax - xoff) / resolution))
        yoff = np.ceil(ymax / resolution) * resolution
        ysize = int(1 - np.floor((ymin - yoff) / resolution))

        # run plyflatten
        dsm = plyflatten(cloud, xoff, yoff, resolution, xsize, ysize, radius=1, sigma=float("inf"))
        #print(f'dsm.shape : {dsm.shape}');  exit(1)
        crs = rasterio_crs(crs_proj("{}{}".format(scene_params[2], scene_params[3]), crs_type="UTM"))

        profile = {}
        profile["dtype"] = dsm.dtype
        profile["height"] = dsm.shape[0]
        profile["width"] = dsm.shape[1]
        profile["count"] = 1
        profile["driver"] = "GTiff"
        profile["nodata"] = float("nan")
        profile["crs"] = crs
        profile["transform"] = affine.Affine(resolution, 0.0, xoff, 0.0, -resolution, yoff)
        #with rasterio.open(os.path.join(base_path, "dsm", name), "w", **profile) as f:
        with rasterio.open(os.path.join(base_path, "dsm", f'{id_image}_dsm.tif'), "w", **profile) as f:
            f.write(dsm[:, :, 0], 1)
        dir_out = os.path.join(base_path, "dsm")
        n_pxl_nan, ratio_nan = save_image(dir_out, f'{id_image}_{idx:02d}', dsm.squeeze())
        print(f'n_pxl_nan : {n_pxl_nan}, ratio_nan : {ratio_nan}\n');
        #exit(1)
    #exit(1)     
    return

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool, opacity_treshold : float = None, resolution: float = 0.5):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

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

        bg_color = [1,0,1,scene.getTrainCameras()[0].altitude_bounds[0].item(),0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, f"train_op{opacity_treshold}", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, [scene.scene_shift, scene.scene_scale, scene.scene_n, scene.scene_l], resolution)

        if not skip_test:
             render_set(dataset.model_path, f"test_op{opacity_treshold}", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, [scene.scene_shift, scene.scene_scale, scene.scene_n, scene.scene_l], resolution)

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
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, opacity_treshold=opacity_treshold, resolution=args.res)
