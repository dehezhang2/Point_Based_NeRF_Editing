# Digital_Human_NeRF_Editing
This is a repository containing the code of the course project of digital human. Our topic is NeRF editing based on point deformation


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