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
import cv2
from sklearn.cluster import KMeans
from scipy.spatial.distance import pdist


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



def calculate_lab_variance(image: np.ndarray) -> float:
    """Lab 색공간에서의 분산을 계산합니다."""
    # 입력이 0-1 범위인 경우 0-255로 변환
    if image.dtype == np.float32 or image.dtype == np.float64:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
    
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    
    variances = []
    for channel in range(3):
        var = np.var(lab_image[:, :, channel])
        variances.append(var)
    
    return np.mean(variances)

def calculate_unique_color_ratio(image: np.ndarray) -> float:
    """전체 픽셀 대비 고유 색상의 비율을 계산합니다."""
    pixels = image.reshape(-1, 3)
    unique_colors = np.unique(pixels, axis=0)
    
    return len(unique_colors) / len(pixels)

def calculate_color_std(image: np.ndarray) -> float:
    """RGB 채널별 표준편차의 평균을 계산합니다."""
    stds = []
    for channel in range(3):
        std = np.std(image[:, :, channel])
        stds.append(std)
    
    return np.mean(stds)

def calculate_cluster_diversity(image: np.ndarray, n_clusters: int = 8) -> float:
    """K-means 클러스터링을 사용하여 색상 다양성을 측정합니다."""
    # 이미지를 1D 배열로 변환
    pixels = image.reshape(-1, 3)
    
    # 샘플링 (계산 속도 향상)
    if len(pixels) > 10000:
        indices = np.random.choice(len(pixels), 10000, replace=False)
        pixels = pixels[indices]
    
    # K-means 클러스터링
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # 클러스터 중심 간의 평균 거리 계산
    centers = kmeans.cluster_centers_
    distances = pdist(centers)
    
    return np.mean(distances)

def calculate_histogram_entropy(image: np.ndarray, bins: int = 64) -> float:
    """히스토그램 엔트로피를 계산합니다."""
    # 각 채널별 히스토그램 계산
    entropies = []
    for channel in range(3):
        hist, _ = np.histogram(image[:, :, channel], bins=bins, range=(0, 256))
        hist = hist / hist.sum()  # 정규화
        hist = hist[hist > 0]  # 0인 값 제거
        entropy = -np.sum(hist * np.log2(hist))
        entropies.append(entropy)
    
    return np.mean(entropies)

def calculate_color_variety_metrics(image: np.ndarray) -> dict[str, float]:
    """
    이미지의 색 분포 다양성을 측정하는 여러 메트릭을 계산합니다.
    
    Args:
        image: 0-1로 정규화된 RGB 또는 BGR 이미지 (H, W, C) 형태
    
    Returns:
        다양한 색 분포 메트릭들을 포함한 딕셔너리
    """
    # 0-1 범위를 0-255 범위로 변환 (계산을 위해)
    image_255 = (image * 255).astype(np.uint8)
    
    # BGR인 경우 RGB로 변환, 이미 RGB인 경우 그대로 사용
    if len(image.shape) == 3 and image.shape[2] == 3:
        # OpenCV 스타일 (BGR)로 가정하고 RGB로 변환
        rgb_image = cv2.cvtColor(image_255, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = image_255
    
    # 1. Color Histogram Entropy (히스토그램 엔트로피)
    hist_entropy = calculate_histogram_entropy(rgb_image)
    
    # 2. Color Clustering Diversity (K-means 클러스터링 기반 다양성)
    cluster_diversity = calculate_cluster_diversity(rgb_image)
    
    # 3. Color Standard Deviation (색상 표준편차)
    color_std = calculate_color_std(rgb_image)
    
    # 4. Unique Color Ratio (고유 색상 비율)
    unique_ratio = calculate_unique_color_ratio(rgb_image)
    
    # 5. Lab Color Space Variance (Lab 색공간에서의 분산)
    lab_variance = calculate_lab_variance(image_255)
    
    return {
        'histogram_entropy': hist_entropy,
        'cluster_diversity': cluster_diversity,
        'color_std': color_std,
        'unique_ratio': unique_ratio,
        'lab_variance': lab_variance
    }

def select_reference_image(images: dict[str, np.ndarray]) -> str:
    """
    여러 이미지 중에서 색 분포가 가장 다양한 이미지를 선택합니다.
    
    Args:
        images: {image_id: image_array} 형태의 딕셔너리
                image_array는 0-1로 정규화된 numpy array (H, W, C)
    
    Returns:
        선택된 이미지의 id (string)
    """
    metrics_results = {}
    
    for image_id, image in images.items():
        if image is None:
            print(f"Warning: Image with id '{image_id}' is None")
            continue
            
        # 색 분포 메트릭 계산
        metrics = calculate_color_variety_metrics(image)
        metrics_results[image_id] = metrics
        
        print(f"\n{image_id}:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")
    
    # 종합 점수 계산 (가중 평균)
    weights = {
        'histogram_entropy': 0.3,
        'cluster_diversity': 0.25,
        'color_std': 0.2,
        'unique_ratio': 0.15,
        'lab_variance': 0.1
    }
    
    best_image_id = None
    best_score = -1
    
    for image_id, metrics in metrics_results.items():
        # 정규화된 점수 계산
        score = sum(weights[metric] * metrics[metric] for metric in weights.keys())
        metrics_results[image_id]['composite_score'] = score
        
        if score > best_score:
            best_score = score
            best_image_id = image_id
    
    print(f"\n=== 선택된 기준 이미지 ===")
    print(f"이미지 ID: {best_image_id}")
    print(f"종합 점수: {best_score:.4f}")
    
    return best_image_id, best_score


def color_transfer_reinhard(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """
    Reinhard 방법을 사용한 색상 전송
    
    Args:
        source: 색상을 가져올 소스 이미지 (기준 이미지) - 0-1 정규화된 이미지
        target: 색상을 적용할 타겟 이미지 - 0-1 정규화된 이미지
    
    Returns:
        색상이 전송된 이미지 (0-1 범위)
    """
    # 0-1 범위를 0-255로 변환
    source_255 = (source * 255).astype(np.uint8)
    target_255 = (target * 255).astype(np.uint8)
    
    # Lab 색공간으로 변환
    source_lab = cv2.cvtColor(source_255, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target_255, cv2.COLOR_BGR2LAB).astype(np.float32)
    
    # 각 채널의 평균과 표준편차 계산
    source_mean = np.mean(source_lab, axis=(0, 1))
    source_std = np.std(source_lab, axis=(0, 1))
    
    target_mean = np.mean(target_lab, axis=(0, 1))
    target_std = np.std(target_lab, axis=(0, 1))
    
    # 색상 전송 수행
    result_lab = target_lab.copy()
    for i in range(3):
        if target_std[i] > 1e-6:  # 0으로 나누기 방지
            result_lab[:, :, i] = (target_lab[:, :, i] - target_mean[i]) * (source_std[i] / target_std[i]) + source_mean[i]
    
    # 값 범위 클리핑
    result_lab = np.clip(result_lab, 0, 255)
    
    # BGR로 다시 변환
    result_255 = cv2.cvtColor(result_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)
    
    # 0-1 범위로 정규화하여 반환
    result = result_255.astype(np.float32) / 255.0
    
    return result


def transfer_color_to_one_of_variety(di_fn_im01):
    fn_ref, skore = select_reference_image(di_fn_im01)
    im01_ref = di_fn_im01[fn_ref]
    for fn, im01_tgt in di_fn_im01.items():
        if fn == fn_ref:
            continue
        im_transferred = color_transfer_reinhard(im01_ref, im01_tgt)
        di_fn_im01[fn] = im_transferred
    return di_fn_im01      
    

def readAffineSceneInfo(path, images, type_norm, shall_color_transfer, eval):
    #aa = bb
    with open(os.path.join(path, 'affine_models.json'),'r') as metadatas:
        metadatas = json.load(metadatas)
    #print(f'len(metadatas) : {len(metadatas)}') 
    #print(f"\neval : {eval}\n"); exit(1)
    cam_infos = []

    v_min_total = 9999999999999999; v_max_total = -v_min_total;
    if 'NORM_DATASET' == type_norm:
        for n, metadata in enumerate(metadatas):
            #print(f'\nmetadata : {metadata}\n');    #exit(1)
            if metadata['img'] == 'Nadir':
                continue
            img_path = os.path.join(images, metadata['img'])
            img = iio.read(img_path)
            min_ori = img.min();    max_ori = img.max()
            if min_ori < v_min_total:
                v_min_total = min_ori
            if max_ori > v_max_total:
                v_max_total = max_ori
                     
        for n, metadata in enumerate(metadatas):
            img_path = os.path.join(images, metadata['img'])
            if metadata['img'] == 'Nadir':
                continue
            img = iio.read(img_path)
            min_ori = img.min();    max_ori = img.max()
            if 0 <= min_ori and 1 < max_ori and max_ori <= 255:
                img = img / 255.0
            elif 255 < max_ori:
                img = (img - v_min_total) / (v_max_total - v_min_total)
            min_norm = img.min();    max_norm = img.max()
            '''
            print(f'\timg.shape : {img.shape}');  #exit(1) #   (815, 746, 3)
            print(f'\tmin_ori : {min_ori}, max_ori : {max_ori}'); #   102 805
            print(f'\tmin_norm : {min_norm}, max_norm : {max_norm}'); #   0 255
            '''
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

    elif 'NORM_DATASET_CHANNEL_WISE' == type_norm:
        channel_min = [9999999999] * 3;  channel_max = [-9999999999] * 3
        #print(f'channel_min : {channel_min}, channel_max : {channel_max}'); exit(1)
        for n, metadata in enumerate(metadatas):
            #print(f'\nmetadata : {metadata}\n');    #exit(1)
            if metadata['img'] == 'Nadir':
                continue
            img_path = os.path.join(images, metadata['img'])
            img = iio.read(img_path)
            for chn in range(3):
                channel = img[:, :, chn]
                c_min = channel.min();  c_max = channel.max()
                if c_min < channel_min[chn]:
                    channel_min[chn] = c_min
                if channel_max[chn] < c_max:
                    channel_max[chn] = c_max
                     
        for n, metadata in enumerate(metadatas):
            img_path = os.path.join(images, metadata['img'])
            if metadata['img'] == 'Nadir':
                continue
            img = iio.read(img_path)
            min_ori = img.min();    max_ori = img.max()
            if 0 <= min_ori and 1 < max_ori and max_ori <= 255:
                img = img / 255.0
            elif 255 < max_ori:
                for chn in range(3):
                    channel = img[:, :, chn]
                    img[:, :, chn] = (channel - channel_min[chn]) / (channel_max[chn] - channel_min[chn])
            min_norm = img.min();    max_norm = img.max()
            '''
            print(f'\timg.shape : {img.shape}');  #exit(1) #   (815, 746, 3)
            print(f'\tmin_ori : {min_ori}, max_ori : {max_ori}'); #   102 805
            print(f'\tmin_norm : {min_norm}, max_norm : {max_norm}'); #   0 255
            '''
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



    elif 'NORM_IMAGE_WISE' == type_norm:
        for n, metadata in enumerate(metadatas):
            img_path = os.path.join(images, metadata['img'])
            if metadata['img'] == 'Nadir':
                continue
          
            img = iio.read(img_path)
            min_ori = img.min();    max_ori = img.max()
            if 0 <= min_ori and 1 < max_ori and max_ori <= 255:
                img = img / 255.0
            elif 255 < max_ori:
                img = (img - min_ori) / (max_ori - min_ori)
            min_norm = img.min();    max_norm = img.max()
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
       
    elif 'NORM_CHANNEL_WISE' == type_norm:
        di_fn_im = {}
        for n, metadata in enumerate(metadatas):
            #print(f"metadata['model'] : {metadata['model']}");   exit(1)
            name_img = metadata['img']
            img_path = os.path.join(images, name_img)
            #print(f"\nimages : {images}, metadata['img'] : {metadata['img']}\n"); exit(1)
            #print(f"img_path : {img_path}"); exit(1)
            #if metadata['img'] != 'Nadir':
            if metadata['img'] == 'Nadir':
                continue
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
            di_fn_im[name_img] = img
        if shall_color_transfer:
            di_fn_im = transfer_color_to_one_of_variety(di_fn_im)
        for n, metadata in enumerate(metadatas):
            name_img = metadata['img']
            if name_img == 'Nadir':
                continue
            img_path = os.path.join(images, name_img)
            img = di_fn_im[name_img]
            '''
            min_norm = img.min();    max_norm = img.max()
            print(f'img_path : {img_path}');  #exit(1) #   ndarray
            print(f'\timg.shape : {img.shape}');  #exit(1) #   (815, 746, 3)
            print(f'\tmin_ori : {min_ori}, max_ori : {max_ori}'); #   102 805
            print(f'\tmin_norm : {min_norm}, max_norm : {max_norm}'); #   0 255
            #exit(1)
            '''
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

        target_density = 0.13 # 0.13 gaussians per true cubic meter.    ori
        #target_density = 0.13 * 0.5 * 0.5 * 0.5 #* 0.5 # 0.13 gaussians per true cubic meter.
        #target_density = 1.3 # 0.13 gaussians per true cubic meter.
        scale = metadata['model']['scale']
        sides = max_world * 1.1 - min_world * 1.1
        volume_inner = np.prod(sides)
        volume_outer = 2**3
        num_pts_to_be_generated = int(target_density * volume_outer * scale**3)
        print(f'# pt : {num_pts_to_be_generated}, target_density : {target_density}, scale : {scale}');   #exit(1)
        # # pt : 423487383, target_density : 0.13, scale : 741.2005151003221 for add_WV3
        # # pt : 18438347, target_density : 0.13, scale : 260.7531437054834 for JAX_214
        # # pt : 76696983, target_density : 0.008125, scale : 1056.7079523446394 for add_EROS
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
