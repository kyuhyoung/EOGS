# EOGS

List of tools and scripts developped while prepping a Gaussian-splatting version of EO-NeRF

## How to install

* Create a conda environment
```bash
conda create -n eogs python=3.10
conda activate eogs
```
* Install pytorch and torchvision
```bash
pip install torch torchvision
```
* Install packages in `requirements.txt` 
```bash
pip install -r requirements.txt
```
* Install 3DGS CUDA kernels
```bash
pip install src/gaussiansplatting/submodules/diff-gaussian-rasterization
pip install src/gaussiansplatting/submodules/simple-knn
```

## How to create dataset
Download the dataset from the GitHub Release page: [dataset_v01](https://github.com/mezzelfo/EOGS/releases/download/dataset_v01/data.zip)

Extract the dataset in the `data` folder (`unzip -q data.zip -d data`). The structure should look like this:
```
EOGS/
├── data/
│   ├── images/
│   │   ├── JAX_004/
│   │   │   ├── JAX_004_006_RGB.tif
│   │   │   ├── JAX_004_007_RGB.json
│   │   │   ...
|   │   ├── JAX_068/
|   │   ...
│   ├── rpcs/
│   │   ├── JAX_004/
│   │   │   ├── JAX_004_006_RGB.json
│   │   │   ├── JAX_004_007_RGB.json
│   │   │   ...
│   │   │   ├── test.txt
│   │   │   └── train.txt
|   │   ├── JAX_068/
|   │   ...
│   ├── truth/
│   │   ├── JAX_004/
│   │   │   ├── JAX_004_CLS.tif
│   │   │   ├── JAX_004_DSM.txt (optional)
│   │   │   └── JAX_004_DSM.tif
|   │   ├── JAX_068/
|   │   ...
│   ├── README.md/
```

Copy the dataset DFC2019 and root_dir in a _datasets_ folder. Then run the following commands to create the affine approximations of the camera models:
```bash
python scripts/dataset_creation/to_affine.py --scene_name JAX_004
python scripts/dataset_creation/to_affine.py --scene_name JAX_068
python scripts/dataset_creation/to_affine.py --scene_name JAX_214
python scripts/dataset_creation/to_affine.py --scene_name JAX_260
python scripts/dataset_creation/to_affine.py --scene_name IARPA_001
python scripts/dataset_creation/to_affine.py --scene_name IARPA_002
python scripts/dataset_creation/to_affine.py --scene_name IARPA_003
```

## How to reproduce the results
Run the following command to reproduce the results of Table 1 in the [paper](https://arxiv.org/pdf/2412.13047) (be aware that different initial random seeds in `src/gaussiansplatting/utils/general_utils.py/safe_state` will lead to potentially different results):
```bash
bash train.sh reproduceMain
```

# FAQ

> [!TIP]
> if `No module named 'torch'` when install: `submodules/diff-gaussian-rasterization`, `pip install --upgrade setuptools wheel packaging`
 
> [!TIP]
> if `KeyError: 'centerofscene_ECEF` while running the code: regenerate the camera models (see [dataset creation](#how-to-create-dataset))

> [!TIP]
> When using uv: if `No module named 'torch'` when install: `submodules/diff-gaussian-rasterization`, `--no-build-isolation` (recommended by the latest uv version)
