import os
from glob import glob
import json
import tyro
import numpy as np
import rpcm
import sklearn.linear_model as lm
from pyproj import Transformer
import utm
from copy import deepcopy
import re
import rasterio

class MyConverter():
    '''
    This class is used to convert from LONLAT to a choosen world coordinate system.
    In the current implementation, the world coordinate system is a normalized ECEF coordinate system.
    '''
    def __init__(self, scene_metadatas) -> None:
        # scene_metadatas = list of metadata dictionaries, one for each image in the scene.
        # We use all the different rpc models to compute the normalization factor for the conversion.
        # In this way, the conversion is the same for all images in the scene.

        self.lonlat2ecef = Transformer.from_crs("epsg:4326", "epsg:4978", always_xy=True)

        vertices_UTM = []
        vertices_UTM_ground = []
        # TODO compute all positions and then call localization once? (can be useful if we ever consider scenes where not all cameras target the same region)
        # This one doesn't work, I need to figure out why
        #for metadata in scene_metadatas:
        #    rpc = rpcm.RPCModel(d=metadata["rpc"], dict_format="rpcm")
        #    width = metadata["width"]
        #    height = metadata["height"]
        #    min_altitude = metadata["min_alt"]
        #    max_altitude = metadata["max_alt"]

        #    a = np.asarray([min_altitude, max_altitude])
        #    lon, lat = rpc.localization(np.asarray([0, width-1]), np.asarray([0, height-1]), a)
        #    x, y, n, l = utm.from_latlon(lat, lon)
        #    vertices_UTM.append(np.stack([x, y, a], axis=-1))
        #
        #    a = np.asarray([0., 0.])
        #    lon, lat = rpc.localization(np.asarray([0, width-1]), np.asarray([0, height-1]), a)
        #    x, y, n, l = utm.from_latlon(lat, lon)
        #    vertices_UTM_ground.append(np.stack([x, y, a], axis=-1))

        #vertices_UTM = np.concatenate(vertices_UTM, axis=0)
        #vertices_UTM_ground = np.concatenate(vertices_UTM_ground, axis=0)
        '''
        print(f'len(scene_metadatas) : {len(scene_metadatas)}');  #exit(1)
        for iM, metadata in enumerate(scene_metadatas):
            print(f'iM : {iM}, metadata.keys() : {metadata.keys()}')    
            #   img, height, width, sun_elevation, sun_azimuth, acquisition_date, geojson, min_alt, rpc, keypoints
        exit(1)     
        '''
        for iM, metadata in enumerate(scene_metadatas):
            print(f'iM : {iM} / {len(scene_metadatas)}')
            #print(f'metadata["rpc"] : {metadata["rpc"]}');  exit(1)
            if isinstance(metadata["rpc"], str):
                #print('aaa')
                rpc = rpcm.rpc_from_rpc_file(metadata["rpc"])
            else:
                #print('bbb')
                rpc = rpcm.RPCModel(d=metadata["rpc"], dict_format="rpcm")
            #exit(1)
            #print(f'rpc.__dict__ : {rpc.__dict__}');  exit(1)
            width_cropped = metadata["width_cropped"]
            height_cropped = metadata["height_cropped"]
            min_altitude = metadata["min_alt"]
            max_altitude = metadata["max_alt"]
            #print(f'\tmin_altitude : {min_altitude}, max_altitude : {max_altitude}'); exit(1)
            #   -29, 73
            for u in [0, width_cropped - 1]:
                for v in [0, height_cropped - 1]:
                    for a in [min_altitude, max_altitude]:
                        lon, lat = rpc.localization(u, v, a)
                        x, y, n, m = utm.from_latlon(lat, lon)
                        #print(f'\tu : {u}, v : {v}, a : {a}, lon : {lon}, lat : {lat}, x : {x}, y : {y}, n : {n}, m : {m}')
                        vertices_UTM.append(np.stack([x, y, a], axis = -1))
                    for a in [0.0]:
                        lon, lat = rpc.localization(u, v, a)
                        x, y, n, m = utm.from_latlon(lat, lon)
                        #print(f'\tu : {u}, v : {v}, a : {a}, lon : {lon}, lat : {lat}, x : {x}, y : {y}, n : {n}, m : {m}')
                        vertices_UTM_ground.append(np.stack([x, y, a], axis = -1))
        #exit(1)
        vertices_UTM = np.array(vertices_UTM)
        vertices_UTM_ground = np.array(vertices_UTM_ground)

        # Store the center of the scene in ECEF and recompute it in LONLAT
        self.centerofscene_UTM = vertices_UTM_ground.mean(axis=0)

        # Store the normalization
        self.shift = self.centerofscene_UTM

        self.n = n
        self.l = m

        # Search for the furthest point between vertices_ECEF and the center of the scene
        max_dist = 0
        for v in vertices_UTM:
            max_dist = max(max_dist, np.linalg.norm(v - self.centerofscene_UTM))
        self.scale = max_dist # Remember to keep the same scale for all axes (as UTM is a euclidean coordinate system)

        # Compute the bounding box of the scene in the world coordinates
        vertices_world = (vertices_UTM - self.shift) / self.scale
        #   for unnormalizing to UTM coords, p_real = p_normal * self.scale + self.shift
        self.min_world = vertices_world.min(axis=0)
        self.max_world = vertices_world.max(axis=0)

        print("shift :", self.shift, "scale :", self.scale)
        print("min_world :", self.min_world)
        print("max_world :", self.max_world)
        print("volume :", np.prod(self.max_world - self.min_world))
        #exit(1)

    def _LONLAT2NormalizedECEF(self, LONLAT):
        X,Y,Z = self.lonlat2ecef.transform(LONLAT[...,0], LONLAT[...,1], LONLAT[...,2])
        ECEF = np.stack([X,Y,Z], axis=-1)
        return (ECEF - self.shift) / self.scale

    def _LONLAT2NormalizedUTM(self, LONLAT):
        # To UTM
        Z = LONLAT[..., 2]
        X, Y, _, _ = utm.from_latlon(LONLAT[..., 1], LONLAT[..., 0])
        UTM = np.stack([X,Y,Z], axis=-1)
        # Return normalized
        return (UTM - self.shift) / self.scale

    def _ECEF2world(self, ECEF):
        normalizedECEF = (ECEF - self.shift) / self.scale
        return np.einsum('ij,...j->...i', self.R, normalizedECEF)

    def LONLAT2world(self, LONLAT):
        # Convert LONLAT to world coordinates
        # LONLAT is a numpy array of shape (N, 3) where N is the number of points
        # The output is a numpy array of shape (N, 3) with the world coordinates
        # Project to utm and normalized
        return self._LONLAT2NormalizedUTM(LONLAT)



def approximate_W2V_affine(
    rpc: rpcm.RPCModel,
    width: int,
    height: int,
    min_altitude: float,
    max_altitude: float,
    converter: MyConverter,
):
    # crate a meshgrid of the image
    # This defines the view/local/intrinsic coordinate system of the current image
    Nu = 31
    Nv = 37
    Na = 29
    u = np.linspace(0, width - 1, Nu)
    v = np.linspace(0, height - 1, Nv)
    a = np.linspace(min_altitude, max_altitude, Na)
    U, V, A = np.meshgrid(u, v, a, indexing = "ij")
    UVA = np.stack([U, V, A], axis = -1)

    # This are the view/local/intrinsic coordinates of the image with U and V in [-1,1] and A in [min_altitude, max_altitude]
    view = (UVA + np.array([0.5, 0.5, 0])) * np.array([1 / width, 1 / height, 1])
    view[..., :2] = view[..., :2] * 2 - 1

    # Now we want to compute a world/global/extrinsic coordinate system for all the images in the scene
    # We do this by computing the lonlat coordinates (this is already a global coordinate system)
    # Then we use a custom converter that should be global, in the sense that it should be the same for all images in the scene.
    LON, LAT = rpc.localization(U.flatten(), V.flatten(), A.flatten())
    LON = LON.reshape(Nu, Nv, Na)
    LAT = LAT.reshape(Nu, Nv, Na)
    world_coords = converter.LONLAT2world(np.stack([LON, LAT, A], axis=-1))

    # Now we learn a linear mapping where:
    # - inputs are the world/global coordinates
    # - outputs are view/local UVA coordinates
    model_W2V = lm.LinearRegression(fit_intercept=True)
    model_W2V.fit(X=world_coords.reshape(-1, 3), y=view.reshape(-1, 3))
    return model_W2V


#### Compute sun direction
def get_dir_vec_from_el_az(elevation_deg, azimuth_deg):
    # convention: elevation is 0 degrees at nadir, 90 at frontal view
    el = np.radians(90 - elevation_deg)
    az = np.radians(azimuth_deg)
    dir_vec = -1.0 * np.array(
        [np.sin(az) * np.cos(el), np.cos(az) * np.cos(el), np.sin(el)]
    )
    # dir_vec = dir_vec / np.linalg.norm(dir_vec)
    return dir_vec


def pipeline(metadata, converter):
    print(f'type(metadata["rpc"]) : {type(metadata["rpc"])}');  #exit(1)    #   dict
    if isinstance(metadata["rpc"], str):
        rpc = rpcm.rpc_from_rpc_file(metadata["rpc"])
    else:
        rpc = rpcm.RPCModel(d = metadata["rpc"], dict_format="rpcm")
    #print(f'type(rpc) : {type(rpc)}');  exit(1) #   rpcm.rpc_model.RPCModel

    #########################################################
    ### Compute the affine approximation of the camera model
    #########################################################
    model_W2V = approximate_W2V_affine(
        rpc = rpc,
        width = metadata["width_cropped"],
        height = metadata["height_cropped"],
        min_altitude = metadata["min_alt"],
        max_altitude = metadata["max_alt"],
        converter = converter,
    )

    A = np.array(model_W2V.coef_)
    b = np.array(model_W2V.intercept_)

    #########################################################
    ### Computing the affine approximation of the sun
    #########################################################
    # First we compute the change of basis M from lonlat to world coordinates
    # and the center of the scene in world coords
    # M, centerofscene_ECEF = converter.computeJacobian()
    # assert np.allclose(M, np.eye(3), atol=1e-7, rtol=1e-7), M
    # assert np.allclose(centerofscene_ECEF, np.zeros(3)), centerofscene_ECEF
    M = np.eye(3)
    centerofscene_UTM = np.zeros(3)


    # Then we compute the sun direction in ECEF
    local_sun_direction = -get_dir_vec_from_el_az(
        elevation_deg = 90 - float(metadata["sun_elevation"]),
        azimuth_deg = float(metadata["sun_azimuth"]),
    )
    sun_dir_ecef = M.T @ local_sun_direction
    sun_dir_ecef = sun_dir_ecef / (A @ sun_dir_ecef)[2]

    # Then we compute the affine model for the sun
    Asun_dir_ecef = A @ sun_dir_ecef
    myM = np.array([[1, 0, -Asun_dir_ecef[0]], [0, 1, -Asun_dir_ecef[1]], [0, 0, 1]])
    sun_A = myM @ A
    sun_b = -sun_A @ centerofscene_UTM + A @ centerofscene_UTM + b

    # Enrich the metadata with the new models
    metadata['centerofscene_UTM'] = centerofscene_UTM.tolist()
    metadata["model"] = {
        "coef_": A.tolist(),
        "intercept_": b.tolist(),
        "scale": converter.scale,
        "n": converter.n,
        "l": converter.l,
        #"rotation": converter.R.tolist(),
        "center": converter.shift.tolist(),
        "min_world": converter.min_world.tolist(),
        "max_world": converter.max_world.tolist(),
    }
    metadata["sun_model"] = {
        "coef_": sun_A.tolist(),
        "intercept_": sun_b.tolist(),
        "sun_dir_ecef": sun_dir_ecef.tolist(),
        "camera_to_sun": myM.tolist(),
    }

    return metadata


def open_json_file(file_path):
    with open(file_path, "r") as f:
        file = json.load(f)
        #print(f'type(file) : {type(file)}');    exit(1) #   dict
        if 'width' in file:
            file['width_ori'] = file['width'];  file['width_cropped'] = file['width'];  file.pop('width');
        if 'height' in file:
            file['height_ori'] = file['height'];  file['height_cropped'] = file['height'];  file.pop('height');
            #print(f'file : {file}');    
        #exit(1)      
    return file


def test(metadatas):
    for i in range(len(metadatas)):
        for j in range(len(metadatas)):
            A = metadatas[i]["model"]["coef_"]
            sundir = metadatas[j]["sun_model"]["sun_dir_ecef"]
            A = np.array(A)
            sundir = np.array(sundir)
            lolz = (A @ sundir)[2]
            assert abs(lolz - 1) < 1e-4, lolz


def parse_rpc_file(file_path):
    """Parse bb.txt to extract RPC metadata."""
    rpc_data = {}
    #aa = bb
    #print(f'file_path : {file_path}');  exit(1)
    with open(file_path, 'r') as f:
        for line in f:
            key, value = line.strip().split(': ')
            if '_COEFF_' in key:
                if key.startswith('LINE_NUM'):
                    if 'LINE_NUM_COEFF' in rpc_data:
                        rpc_data['LINE_NUM_COEFF'].append(float(value))
                    else:
                        rpc_data['LINE_NUM_COEFF'] = [float(value)]
                elif key.startswith('LINE_DEN'):
                    if 'LINE_DEN_COEFF' in rpc_data:
                        rpc_data['LINE_DEN_COEFF'].append(float(value))
                    else:
                        rpc_data['LINE_DEN_COEFF'] = [float(value)]
                elif key.startswith('SAMP_NUM'):
                    if 'SAMP_NUM_COEFF' in rpc_data:
                        rpc_data['SAMP_NUM_COEFF'].append(float(value))
                    else:
                        rpc_data['SAMP_NUM_COEFF'] = [float(value)]
                elif key.startswith('SAMP_DEN'):
                    if 'SAMP_DEN_COEFF' in rpc_data:
                        rpc_data['SAMP_DEN_COEFF'].append(float(value))
                    else:
                        rpc_data['SAMP_DEN_COEFF'] = [float(value)]
            else:
                # Handle numeric values (remove units like 'pixels', 'degrees', 'meters')
                value = re.sub(r'\s*(pixels|degrees|meters)', '', value)
                rpc_data[key] = float(value)
    #print(f'rpc_data : {rpc_data}');    exit(1)
    return rpc_data


def parse_3dmet_file(file_path):
    """Parse aa.txt to extract RPC metadata and Sun data for each group."""
    groups = {}
    current_group = None
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        i = 0;  iG = 0
        while i < len(lines):
            line = lines[i].strip()
            if line.startswith('BEGIN_GROUP='):
                current_group = line.split('=')[1].strip(';')
                groups[current_group] = {}
                i += 1
                while i < len(lines) and not lines[i].startswith('END_GROUP'):
                    line = lines[i].strip()
                    if '=' in line and not line.startswith(('LineNumCoeff', 'LineDenCoeff', 'sampNumCoeff', 'sampDenCoeff')):
                        key, value = line.split(' = ', 1)
                        value = value.strip(';')
                        try:
                            groups[current_group][key] = float(value)
                        except ValueError:
                            groups[current_group][key] = value
                        current_key = None        
                    elif line.startswith(('LineNumCoeff', 'LineDenCoeff', 'sampNumCoeff', 'sampDenCoeff')):
                        current_key = line.split('= (')[0].strip()
                        groups[current_group][current_key] = []

                    elif current_key and line.endswith(');'): 
                        line = line.rstrip(');').strip()
                        if line:
                            groups[current_group][current_key].append(float(line))
                    elif current_key and line.endswith(','):  
                        line = line.rstrip(',').strip()
                        if line:
                            groups[current_group][current_key].append(float(line))
                    i += 1
                if 'width' in groups[current_group]:
                    groups[current_group]['width_ori'] = groups[current_group]['width']
                    groups[current_group].pop('width')
                if 'height' in groups[current_group]:
                    groups[current_group]['height_ori'] = groups[current_group]['height']
                    groups[current_group].pop('height')
                #print(f'groups[current_group] : {groups[current_group]}');  exit(1)
                if 30 == iG:
                    print(f'groups[current_group] : {groups[current_group]}');  #exit(1)
                iG += 1    
            i += 1
    #exit(1)   
    return groups


def match_rpc(bb_data, aa_groups):
    """Match bb.txt RPC data with aa.txt groups based on LINE_NUM_COEFF and SAMP_NUM_COEFF."""
    bb_line_num_coeff = bb_data.get('LINE_NUM_COEFF', [])
    bb_samp_num_coeff = bb_data.get('SAMP_NUM_COEFF', [])
    #print(f'bb_line_num_coeff : {bb_line_num_coeff}');    #exit(1)
    #print(f'bb_samp_num_coeff : {bb_samp_num_coeff}');    exit(1)
   
    name_matched = None;    data_matched = None      
    dif_min = 10000000000000
    iG = 0; found_G = -1;
    for group_name, group_data in aa_groups.items():
        aa_line_num_coeff = group_data.get('LineNumCoeff', [])
        aa_samp_num_coeff = group_data.get('sampNumCoeff', [])
        #print(f'aa_line_num_coeff : {aa_line_num_coeff}');    #exit(1)
        #print(f'aa_samp_num_coeff : {aa_samp_num_coeff}');    exit(1)
        # Compare coefficients with a small tolerance for floating-point precision
        if len(bb_line_num_coeff) == len(aa_line_num_coeff) and len(bb_samp_num_coeff) == len(aa_samp_num_coeff):
            sum_dif = 0
            for b, a in zip(bb_line_num_coeff, aa_line_num_coeff):
                dif = abs(b - a);   sum_dif += dif
            for b, a in zip(bb_samp_num_coeff, aa_samp_num_coeff):
                dif = abs(b - a);   sum_dif += dif
            print(f'iG : {iG}, sum_dif : {sum_dif}')
            if sum_dif < dif_min:
                dif_min = sum_dif
                name_matched = group_name;  data_matched = group_data
                found_G = iG
            '''         
            all(abs(b - a) < 1e-6 for b, a in zip(bb_line_num_coeff, aa_line_num_coeff)) and all(abs(b - a) < 1e-6 for b, a in zip(bb_samp_num_coeff, aa_samp_num_coeff))):
            name_matched = group_name;  data_matched = group_data
            '''  
        iG += 1      
    print(f'found_G : {found_G}, name_matched : {name_matched}'); #exit(1)
    return name_matched, data_matched


def evaluate_rpc_polynomial(P_n, L_n, H_n, coeffs):
    """Evaluate a third-degree RPC polynomial with 20 coefficients."""
    terms = [
        1, L_n, P_n, H_n, L_n*P_n, L_n*H_n, P_n*H_n, L_n**2, P_n**2, H_n**2,
        L_n*P_n*H_n, L_n**3, L_n*P_n**2, L_n*H_n**2, L_n**2*P_n, P_n**3, P_n*H_n**2,
        L_n**2*H_n, P_n**2*H_n, H_n**3
    ]
    #print(f'len(terms) : {len(terms)}, terms : {terms}');  #exit(1)
    #print(f'len(coeffs) : {len(coeffs)}, coeffs : {coeffs}');  exit(1)
    return sum(c * t for c, t in zip(coeffs, terms))

def compute_image_coordinates(lat, lon, height, rpc_data):
    """Compute normalized image coordinates (r_n, c_n) using RPC model."""
    # Normalize inputs
    P_n = (lat - rpc_data['LatOffset']) / rpc_data['LatScale']
    L_n = (lon - rpc_data['LongOffset']) / rpc_data['LongScale']
    H_n = (height - rpc_data['HeightOffset']) / rpc_data['HeightScale']
    #print(f'lat : {lat}, lon : {lon}, height : {height}');  exit(1) #   39.05, 125.96, -191
    #print(f'P_n : {P_n}, L_n : {L_n}, H_n : {H_n}');  exit(1) #   0.775, -0.9778, -1.0

    #print(f'rpc_data : {rpc_data}');    exit(1)
    # Evaluate polynomials
    line_num = evaluate_rpc_polynomial(P_n, L_n, H_n, rpc_data['LineNumCoeff'])
    line_den = evaluate_rpc_polynomial(P_n, L_n, H_n, rpc_data['LineDenCoeff'])
    samp_num = evaluate_rpc_polynomial(P_n, L_n, H_n, rpc_data['sampNumCoeff'])
    samp_den = evaluate_rpc_polynomial(P_n, L_n, H_n, rpc_data['sampDenCoeff'])
    #print(f'lat : {lat}, lon : {lon}, height : {height}, \tline_num : {line_num}, line_den : {line_den}, samp_num : {samp_num}, samp_den : {samp_den}');  #exit(1) #  0, 0, 0, 0
    
    r_n = line_num / line_den if line_den != 0 else 0
    c_n = samp_num / samp_den if samp_den != 0 else 0

    t_line = r_n * rpc_data['LineScale'];   t_samp = c_n * rpc_data['SampleScale']
    #print(f't_line : {t_line}, t_samp : {t_samp}'); exit(1)

    # Denormalize to image coordinates
    line = r_n * rpc_data['LineScale'] + rpc_data['LineOffset']
    samp = c_n * rpc_data['SampleScale'] + rpc_data['SampleOffset']
    
    return line, samp


def find_height(lat, lon, target_line, target_samp, rpc_data, height_range=(-1000, 1000), step=1):
    """Find the height that maps (lat, lon) to (target_line, target_samp)."""
    min_error = float('inf')
    best_height = None
    #print(f'height_range : {height_range}, step : {step}'); exit(1) #   (-191, 611), 1
    for height in np.arange(height_range[0], height_range[1], step):
        computed_line, computed_samp = compute_image_coordinates(lat, lon, height, rpc_data)
        #print(f'computed_line : {computed_line}, computed_samp : {computed_samp}, target_line : {target_line}, target_samp : {target_samp}');  exit(1)
        error = ((computed_line - target_line) ** 2 + (computed_samp - target_samp) ** 2) ** 0.5
        #print(f'height : {height}, error : {error}');  #exit(1)
        if error < min_error:
            min_error = error
            best_height = height
    #print(f'min_error : {min_error}, best_height : {best_height}'); exit(1)     
    return best_height


def compute_altitude_range(group_data):
    """Compute min and max altitude for a group using corner coordinates."""
    # Define corners and their corresponding image coordinates
    corners = [
        ('UpperLeft', group_data['UpperLeftLat'], group_data['UpperLeftLon'], 0, 0),
        ('UpperRight', group_data['UpperRightLat'], group_data['UpperRightLon'], 0, group_data['width_ori']),
        ('LowerLeft', group_data['LowerLeftLat'], group_data['LowerLeftLon'], group_data['height_ori'], 0),
        ('LowerRight', group_data['LowerRightLat'], group_data['LowerRightLon'], group_data['height_ori'], group_data['width_ori'])
    ]
    #print(f'corners : {corners}');  exit(1)
    #   (39.053, 125.967, 0, 0), (39.056, 126.087, 0, 34767), (38.960, 125.970, 34250, 0), (38.963, 126.090, 34250, 34767)  
    # Estimate height range based on HeightOffset and HeightScale
    height_min = group_data['HeightOffset'] - group_data['HeightScale']
    height_max = group_data['HeightOffset'] + group_data['HeightScale']
    
    heights = []
    for corner_name, lat, lon, target_line, target_samp in corners:
        height = find_height(lat, lon, target_line, target_samp, group_data, (height_min, height_max))
        heights.append(height)
        print(f"corner_name : {corner_name}, Height = {height:.2f} meters")
    #exit(1)
    return min(heights), max(heights)

def get_exact_file_name_from_path(str_path):
    return os.path.splitext(os.path.basename(str_path))[0]


def get_meta_from_rpc_and_3dmet(li_path_rpc, path_3dmet, dir_img):
    meta_groups = parse_3dmet_file(path_3dmet)
    metadatas = []
    for iR, path_rpc in enumerate(li_path_rpc):
        id_img = get_exact_file_name_from_path(path_rpc)
        print(f'iR : {iR}, {path_rpc}')
        di_meta = {'rpc' : path_rpc}
        rpc_data = parse_rpc_file(path_rpc)
        matched_group, matched_data = match_rpc(rpc_data, meta_groups)
        #print(f'matched_data : {matched_data}');    exit(1)
        if matched_group:
            #exit(1)
            min_height, max_height = compute_altitude_range(matched_data)
            #'''
            dif_meter = max_height - min_height;
            print(f'dif_meter : {dif_meter}');  #exit(1) 
            #   dif_meter : 228.99999999999997  for EROS
            #   dif_meter : 1                   for WV3
            # 
            if dif_meter > 10:
                avg_height = 0.5 * (min_height + max_height)
                min_height = avg_height - 5
                max_height = avg_height + 5
            #'''
            di_meta['img'] = f'{id_img}.tif'
            di_meta['min_alt'] = min_height;   di_meta['max_alt'] = max_height
            di_meta['sun_azimuth'] = matched_data.get('SunAzimuth')
            di_meta['sun_elevation'] = matched_data.get('SunElevation')
            path_img = os.path.join(dir_img, di_meta['img'])
            with rasterio.open(path_img) as img:
                di_meta['width_cropped'] = img.width
                di_meta['height_cropped'] = img.height
            #print(f'di_meta : {di_meta}');  exit(1)      
            '''
            lon_lat_aoi = 
            [
                [
                    [matched_data.get('UpperLeftLon'), matched_data.get('UpperLeftLat')] 
                    [matched_data.get('UpperRightLon'), matched_data.get('UpperRightLat')] 
                    [matched_data.get('LowerRightLon'), matched_data.get('LowerRightLat')] 
                    [matched_data.get('LowerLeftLon'), matched_data.get('LowerLeftLat')] 
                ]
            ]
            di_meta['geojson'] = {'coordinate': lon_lat_aoi}
            print(f"Matched group: {matched_group}")
            print(f"Sun Azimuth: {sun_azimuth} degrees")
            print(f"Sun Elevation: {sun_elevation} degrees")
            '''
            #print(f'matched_data : {matched_data}');    exit(1) 
            metadatas.append(di_meta)
            meta_groups.pop(matched_group)
        else:
            print("No matching group found in aa.txt for the RPC data in bb.txt")
    #exit(1)
    return metadatas



def main(
    root_dir: str = "data/rpcs",
    scene_name: str = "JAX_068",
    dataset_destination_path: str = "data/affine_models",
    dir_imgs: str = "data/images",
):
    #assert scene_name in ["JAX_004", "JAX_068", "JAX_214", "JAX_260", "IARPA_001", "IARPA_002", "IARPA_003"]
    SCENE_METADATA = os.path.expanduser(
        os.path.join(root_dir, scene_name)
    )
    DATASET_DESTINATION = os.path.expanduser(
        os.path.join(dataset_destination_path, scene_name)
    )
    DIR_IMG = os.path.expanduser(os.path.join(dir_imgs, scene_name))

    # Read the scene and for each image:
    # 1. Open the corresponding .json metadata file
    # 2. Run the conversion pipeline
    '''
    #t0 = glob(f"{SCENE_METADATA}/*.json")   #   None
    t0 = f"{SCENE_METADATA}"
    print(f't0 : {t0}');    exit(1)
    '''
    #print(f'SCENE_METADATA : {SCENE_METADATA}');    exit(1) #   /workspace/EOGS/data/rpcs/JAX_068
    if scene_name.startswith('add'):
        metadatas = []
        li_path_rpc = sorted(glob(f"{SCENE_METADATA}/*.rpc"))
        #print(f'li_path_rpc : {li_path_rpc}');  exit(1)   
        #   li_path_rpc : ['/workspace/EOGS/data//rpcs/add_WV3/PanSharpen1.rpc', '/workspace/EOGS/data//rpcs/add_WV3/PanSharpen2.rpc']
        li_path_meta = glob(f"{SCENE_METADATA}/*.3DMET")
        if not li_path_meta:
            print(f'There is NO 3DMET file under {SCENE_METADATA}. Check the directory');    exit(1)
        path_3dmet = li_path_meta[0]
        #   for all 'rpc' file in the folder
        metadatas = get_meta_from_rpc_and_3dmet(li_path_rpc, path_3dmet, DIR_IMG)
        '''
        for path_rpc in li_path_rpc:
            #   append rpc dict to list
            print(f'path_rpc : {path_rpc}, path_3dmet : {path_3dmet}');  exit(1)   
            di_meta = get_meta_from_rpc_and_3dmet(path_rpc, path_3dmet)
            #di_rpc = {'rpc' : path_rpc}
            metadatas.append(di_meta)
        '''
    else:
        metadatas = sorted(glob(f"{SCENE_METADATA}/*.json"))
        #print(f'type(metadatas) : {type(metadatas)}');  exit(1);    #   list
        metadatas = map(open_json_file, metadatas)
        #print(f'type(metadatas) : {type(metadatas)}');  exit(1) #   map
        metadatas = list(metadatas)
        #print(f'type(metadatas[0]) : {type(metadatas[0])}');  exit(1) #   dict
    #print(f'metadatas : {metadatas}');  exit(1)
    T = MyConverter(metadatas)
    metadatas = map(lambda m: pipeline(m, T), metadatas)
    metadatas = list(metadatas)
    '''
    for metadata in metadatas:
        print(f'type(metadata) : {type(metadata)}');    # dict
    exit(1)     
    '''
    # Generate a metadata for a perfectly nadir camera
    metadata = deepcopy(metadatas[0])
    metadata["img"] = "Nadir"
    metadata["model"]["coef_"] = [[0., 1., 0.], [1., 0., 0.], [0., 0., metadata["model"]["scale"]]]
    metadata["model"]["intercept_"] = [0., 0., 0.]

    metadatas.append(metadata)

    # Now that the new metadata has been computed, we run a few tests
    test(metadatas)

    # Finally, we save the new metadata
    #print(f'DATASET_DESTINATION : {DATASET_DESTINATION}');  exit(1) 
    os.makedirs(DATASET_DESTINATION, exist_ok=True)
    with open(f"{DATASET_DESTINATION}/affine_models.json", "w") as f:
        json.dump(metadatas, f, indent=4)
    # Copy also the test.txt and train.txt files
    os.system(f"cp {SCENE_METADATA}/test.txt {DATASET_DESTINATION}")
    os.system(f"cp {SCENE_METADATA}/train.txt {DATASET_DESTINATION}")
    print("Finished 'to_affine'")

if __name__ == "__main__":
    tyro.cli(main)
