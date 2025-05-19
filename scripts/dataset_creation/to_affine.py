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

        for metadata in scene_metadatas:
            rpc = rpcm.RPCModel(d=metadata["rpc"], dict_format="rpcm")
            width = metadata["width"]
            height = metadata["height"]
            min_altitude = metadata["min_alt"]
            max_altitude = metadata["max_alt"]
            for u in [0, width-1]:
                for v in [0, height-1]:
                    for a in [min_altitude, max_altitude]:
                        lon, lat = rpc.localization(u, v, a)
                        x, y, n, m = utm.from_latlon(lat, lon)
                        vertices_UTM.append(np.stack([x, y, a], axis=-1))
                    for a in [0.0]:
                        lon, lat = rpc.localization(u, v, a)
                        x, y, n, m = utm.from_latlon(lat, lon)
                        vertices_UTM_ground.append(np.stack([x, y, a], axis=-1))
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
        self.min_world = vertices_world.min(axis=0)
        self.max_world = vertices_world.max(axis=0)

        print("shift", self.shift, "scale", self.scale)
        print("min_world", self.min_world)
        print("max_world", self.max_world)
        print("volume", np.prod(self.max_world - self.min_world))

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
    u = np.linspace(0, width-1, Nu)
    v = np.linspace(0, height-1, Nv)
    a = np.linspace(min_altitude, max_altitude, Na)
    U, V, A = np.meshgrid(u, v, a, indexing="ij")
    UVA = np.stack([U, V, A], axis=-1)

    # This are the view/local/intrinsic coordinates of the image with U and V in [-1,1] and A in [min_altitude, max_altitude]
    view = (UVA+np.array([0.5,0.5,0])) * np.array([1 / width, 1 / height, 1])
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
    rpc = rpcm.RPCModel(d=metadata["rpc"], dict_format="rpcm")

    #########################################################
    ### Compute the affine approximation of the camera model
    #########################################################
    model_W2V = approximate_W2V_affine(
        rpc=rpc,
        width=metadata["width"],
        height=metadata["height"],
        min_altitude=metadata["min_alt"],
        max_altitude=metadata["max_alt"],
        converter=converter,
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
        elevation_deg=90 - float(metadata["sun_elevation"]),
        azimuth_deg=float(metadata["sun_azimuth"]),
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


def main(
    root_dir: str = "data/rpcs",
    scene_name: str = "JAX_068",
    dataset_destination_path: str = "data/affine_models",
):
    assert scene_name in ["JAX_004", "JAX_068", "JAX_214", "JAX_260", "IARPA_001", "IARPA_002", "IARPA_003"]
    SCENE_METADATA = os.path.expanduser(
        os.path.join(root_dir, scene_name)
    )
    DATASET_DESTINATION = os.path.expanduser(
        os.path.join(dataset_destination_path, scene_name)
    )

    # Read the scene and for each image:
    # 1. Open the corresponding .json metadata file
    # 2. Run the conversion pipeline
    metadatas = sorted(glob(f"{SCENE_METADATA}/*.json"))
    metadatas = map(open_json_file, metadatas)
    metadatas = list(metadatas)
    T = MyConverter(metadatas)
    metadatas = map(lambda m: pipeline(m, T), metadatas)
    metadatas = list(metadatas)

    # Generate a metadata for a perfectly nadir camera
    metadata = deepcopy(metadatas[0])
    metadata["img"] = "Nadir"
    metadata["model"]["coef_"] = [[0., 1., 0.], [1., 0., 0.], [0., 0., metadata["model"]["scale"]]]
    metadata["model"]["intercept_"] = [0., 0., 0.]

    metadatas.append(metadata)

    # Now that the new metadata has been computed, we run a few tests
    test(metadatas)

    # Finally, we save the new metadata
    os.makedirs(DATASET_DESTINATION, exist_ok=True)
    with open(f"{DATASET_DESTINATION}/affine_models.json", "w") as f:
        json.dump(metadatas, f, indent=4)
    # Copy also the test.txt and train.txt files
    os.system(f"cp {SCENE_METADATA}/test.txt {DATASET_DESTINATION}")
    os.system(f"cp {SCENE_METADATA}/train.txt {DATASET_DESTINATION}")


if __name__ == "__main__":
    tyro.cli(main)
