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
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import iio
from scene.cameras import AffineCameraInfo

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    image: np.array
    image_path: str
    image_name: str
    width: int
    height: int

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center

    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, images_folder):
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        image_path = os.path.join(images_folder, os.path.basename(extr.name))
        image_name = os.path.basename(image_path).split(".")[0]
        image = Image.open(image_path)

        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                              image_path=image_path, image_name=image_name, width=width, height=height)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'f4'), ('green', 'f4'), ('blue', 'f4')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, eval, llffhold=8):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except Exception:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    reading_dir = "images" if images is None else images
    cam_infos_unsorted = readColmapCameras(cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, images_folder=os.path.join(path, reading_dir))
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    if eval:
        train_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold != 0]
        test_cam_infos = [c for idx, c in enumerate(cam_infos) if idx % llffhold == 0]
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    if not os.path.exists(ply_path):
        print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
        try:
            xyz, rgb, _ = read_points3D_binary(bin_path)
        except Exception:
            xyz, rgb, _ = read_points3D_text(txt_path)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readCamerasFromTransforms(path, transformsfile, white_background, extension=".png"):
    cam_infos = []

    with open(os.path.join(path, transformsfile)) as json_file:
        contents = json.load(json_file)
        fovx = contents["camera_angle_x"]

        frames = contents["frames"]
        for idx, frame in enumerate(frames):
            cam_name = os.path.join(path, frame["file_path"] + extension)

            # NeRF 'transform_matrix' is a camera-to-world transform
            c2w = np.array(frame["transform_matrix"])
            # change from OpenGL/Blender camera axes (Y up, Z back) to COLMAP (Y down, Z forward)
            c2w[:3, 1:3] *= -1

            # get the world-to-camera transform and set R, T
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3,:3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = os.path.join(path, cam_name)
            image_name = Path(cam_name).stem
            image = Image.open(image_path)

            im_data = np.array(image.convert("RGBA"))

            bg = np.array([1,1,1]) if white_background else np.array([0, 0, 0])

            norm_data = im_data / 255.0
            arr = norm_data[:,:,:3] * norm_data[:, :, 3:4] + bg * (1 - norm_data[:, :, 3:4])
            image = Image.fromarray(np.array(arr*255.0, dtype=np.byte), "RGB")

            fovy = focal2fov(fov2focal(fovx, image.size[0]), image.size[1])
            FovY = fovy
            FovX = fovx

            cam_infos.append(CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, image=image,
                            image_path=image_path, image_name=image_name, width=image.size[0], height=image.size[1]))

    return cam_infos

def readNerfSyntheticInfo(path, white_background, eval, extension=".png"):
    print("Reading Training Transforms")
    train_cam_infos = readCamerasFromTransforms(path, "transforms_train.json", white_background, extension)
    print("Reading Test Transforms")
    test_cam_infos = readCamerasFromTransforms(path, "transforms_test.json", white_background, extension)

    if not eval:
        train_cam_infos.extend(test_cam_infos)
        test_cam_infos = []

    nerf_normalization = getNerfppNorm(train_cam_infos)

    ply_path = os.path.join(path, "points3d.ply")
    if not os.path.exists(ply_path):
        # Since this data set has no colmap data, we start with random points
        num_pts = 100_000
        print(f"Generating random point cloud ({num_pts})...")

        # We create random points inside the bounds of the synthetic Blender scenes
        xyz = np.random.random((num_pts, 3)) * 2.6 - 1.3
        shs = np.random.random((num_pts, 3)) / 255.0
        pcd = BasicPointCloud(points=xyz, colors=SH2RGB(shs), normals=np.zeros((num_pts, 3)))

        storePly(ply_path, xyz, SH2RGB(shs) * 255)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info

def readAffineSceneInfo(path, images, eval):
    #aa = bb
    with open(os.path.join(path, 'affine_models.json'),'r') as metadatas:
        metadatas = json.load(metadatas)
    #print(f'len(metadatas) : {len(metadatas)}') 
    #print(f"\neval : {eval}\n"); exit(1)
    cam_infos = []
    for n, metadata in enumerate(metadatas):
        #print(f"metadata['model'] : {metadata['model']}");   exit(1)
        img_path = os.path.join(images, metadata['img'])
        #print(f"\nimages : {images}, metadata['img'] : {metadata['img']}\n"); exit(1)
        #print(f"img_path : {img_path}"); exit(1)
        if metadata['img'] != 'Nadir':
            img = iio.read(img_path)
            min_ori = img.min();    max_ori = img.max()
            if 0 <= min_ori and 1 < max_ori and max_ori <= 255:
                img = img / 255.0
            elif  255 < max_ori:
                #img = (img - min_ori) / (max_ori - min_ori) 
                #normalized = np.zeros_like(image_array, dtype=np.float32)
                for c in range(3):
                    channel = img[:, :, c]
                    c_min = channel.min();    c_max = channel.max()
                    img[:, :, c] = (channel - c_min) / (c_max - c_min)

            '''
            if 'SYNEW' not in img_path:
                print('DDD')
                img = img / 255.0
            '''    
        min_norm = img.min();    max_norm = img.max()
        print(f'img_path : {img_path}');  #exit(1) #   ndarray
        #print(f'type(img) : {type(img)}');  #exit(1) #   ndarray
        print(f'\timg.shape : {img.shape}');  #exit(1) #   (815, 746, 3)
        print(f'\tmin_ori : {min_ori}, max_ori : {max_ori}'); #   102 805
        print(f'\tmin_norm : {min_norm}, max_norm : {max_norm}'); #   0 255
        #exit(1)
        if '-NEW-' in img_path:
            reference_altitude = images.replace('-NEW-', '-SYNEW-')
        else:
            #assert '-SYNEW-' in img_path, "Reference altitude not found"
            reference_altitude = images
        reference_altitude = os.path.join(reference_altitude, 'altitude', metadata['img'])
        if not os.path.exists(reference_altitude):
            reference_altitude = img.copy()[...,0]*0.0
            print("Warning: Reference altitude not found, using zeros")
        else:
            reference_altitude = np.squeeze(iio.read(reference_altitude))

        lm_coef_ = np.array(metadata['model']['coef_'])
        lm_intercept_ = np.array(metadata['model']['intercept_'])
        sun_lm_coef_ = np.array(metadata['sun_model']['coef_'])
        sun_lm_intercept_ = np.array(metadata['sun_model']['intercept_'])
        altitude_bounds = np.array([metadata['min_alt'], metadata['max_alt']])
        min_world = np.array(metadata['model']['min_world'])
        max_world = np.array(metadata['model']['max_world'])

        caminfo = AffineCameraInfo(
            uid=None,
            R=None,
            T=None,
            FovY=None,
            FovX=None,
            image=img,
            is_reference_camera=False,
            reference_altitude=reference_altitude,
            image_path=img_path,
            image_name=metadata['img'].replace('.tif',''),
            width = metadata['width_cropped'],
            height = metadata['height_cropped'],
            centerofscene_ECEF=np.array(metadata['centerofscene_UTM']),
            affine_coef=lm_coef_,
            affine_inter=lm_intercept_,
            altitude_bounds=altitude_bounds,
            min_world = min_world,
            max_world = max_world,
            sun_affine_coef=sun_lm_coef_,
            sun_affine_inter=sun_lm_intercept_,
            camera_to_sun=np.array(metadata['sun_model']['camera_to_sun'])
        )

        cam_infos.append(caminfo)
    
    if eval:
        with open(os.path.join(path, 'train.txt'), 'r') as trainsplit:
            trainsplit = trainsplit.read().splitlines()
            trainsplit = [x.replace('.json','') for x in trainsplit]
        with open(os.path.join(path, 'test.txt'), 'r') as testsplit:
            testsplit = testsplit.read().splitlines()
            testsplit = [x.replace('.json','') for x in testsplit]
        train_cam_infos = []
        test_cam_infos = []
        for caminfo in cam_infos[:-1]:
            if caminfo.image_name in trainsplit:
                train_cam_infos.append(caminfo)
            elif caminfo.image_name in testsplit:
                test_cam_infos.append(caminfo)
            else:
                raise RuntimeError("Image not in train or test split!")

        # add the last camera (perfectly nadir) to the test
        test_cam_infos.append(cam_infos[-1])
    else:
        train_cam_infos = cam_infos
        test_cam_infos = []

    # Setting the first train camera as the reference camera
    train_cam_infos[0].is_reference_camera = True

    ply_path = os.path.join(path, "points3d.ply")
    if True: #not os.path.exists(ply_path):
        # In the to_affine.py script we normalized the scene so that:
        # - it is strictly inside [-1,1]^3
        # - it is euclidean (as ECEF) and it unit of measures are meters/metadata['model']['scale']
        # - the box is aligned with longitude and latitude
        # Moreover metadata['model']['min_world'] and metadata['model']['max_world']
        # contain the actual bounds of the scene in [-1,1]^3

        # We now generate random points inside the "inner" bounds of the scene keeping a 10% margin
        # It is IMPORTANT to have a truly uniform distribution inside the inner bbox,
        # meaning that the denisty should be isotropic and constant.
        # Otherwise the initialization of the scales won't work properly.

        # To do so, we start with a uniform distribution in [-1,1]^3 and
        # we keep only the points that are inside the inner bbox.
        # We aim for a target density (expressed in gaussians per true cubic meter).
        # As the distribution is uniform, the total points that should be generated is:
        # N = rho_target * V_out
        # This ensures that the density (both in the inner bbox and in the outer bbox) is correct.
        # There is a catch: we are working in normalized UTM coordinates by a scale factor.

        #target_density = 0.13 # 0.13 gaussians per true cubic meter.
        target_density = 1.3 # 0.13 gaussians per true cubic meter.
        scale = metadata['model']['scale']
        sides = max_world * 1.1 - min_world * 1.1
        volume_inner = np.prod(sides)
        volume_outer = 2**3
        num_pts_to_be_generated = int(target_density * volume_outer * scale**3)
        #print(f'volume_outer : {volume_outer}, scale : {scale}');   exit(1)
        xyz = np.random.rand(num_pts_to_be_generated, 3) * 2 - 1
        inside = np.all(xyz > min_world * 1.1, axis = 1) & np.all(xyz < max_world * 1.1, axis = 1)
        xyz = xyz[inside]
        print("Number of points generated inside the inner bbox:", len(xyz))
        print("Volume inner bbox:", volume_inner)
        print("Density inside the inner bbox:", len(xyz)/(volume_inner * scale**3))
        print("Total density:", num_pts_to_be_generated/(volume_outer * scale**3))
        print("Expected density:", target_density)

        rgb = np.full((len(xyz), 3), 1.1)

        print(rgb)
        storePly(ply_path, xyz, rgb)
    try:
        pcd = fetchPly(ply_path)
    except Exception:
        pcd = None

    radius = np.linalg.norm(xyz - xyz.mean(axis=0), axis=1)
    radius = np.max(radius)*2

    # The radius variable will be used for densification strategies but also for scaling the spatial_lr
    # 100 è troppo
    # 10 è ok
    # 1 è ok
    # radius = radius * 10

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization={'radius': radius, 'scale' : metadata['model']['scale'], 'center': metadata['model']['center'], 'n': metadata['model']['n'], 'l': metadata['model']['l']},
                           ply_path=ply_path)

    return scene_info


sceneLoadTypeCallbacks = {
    "Affine": readAffineSceneInfo,
    "Colmap": readColmapSceneInfo,
    "Blender" : readNerfSyntheticInfo
}
