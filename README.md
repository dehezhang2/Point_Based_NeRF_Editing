# Digital_Human_NeRF_Editing
This is a repository containing the code of the course project of digital human. Our topic is NeRF editing based on point deformation.

## Install Environment
### requirement: (on autodl) 
- Mirror: Python 3.8 (ubuntu20.04) Cuda 11.8
- Install point-NeRF environment
    ```
    conda env create -f environment_autodl.yml
    ```
- Install pytorch3d
    ```
    pip install fvcore iopath
    pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py39_cu117_pyt1131/download.html
    ```
## Run Deformation

### Generate static point cloud
- Put human data in ./pointnerf/data_src/nerf/nerf_synthetic/
- Run point-NeRF on static train dataset
    ```
    cd pointnerf
    bash dev_scripts/w_n360/human_cuda.sh 
    ```
### Deform point cloud
- Run Dynamic Point Field & point-NeRF on test dataset
    ```
    cd pointnerf
    bash dev_scripts/w_n360/human_deform.sh
    ```
    _note: the deformation code ```./pointnerf/run/deform.py``` and ```./pointnerf/dev_scripts/w_n360/human_deform.sh``` are modified based on ```./pointnerf/dev_scripts/w_n360/chair_test.sh``` and ```/pointnerf/run/test_ft.py```. If report error, please check the environment file ```./environment_autodl_export.yml```_