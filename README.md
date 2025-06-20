# Holistic Uncertainty Estimation For Open-Set Recognition
This repository contains implemetation of the methods described in the paper.   
## Repository Setup Steps

1. **Downloading datasets and model weighs**:  
You can [download](https://drive.google.com/file/d/1VUd2XvKFZJiDcO73uHBDTCkqTRD6I2BH/view?usp=sharing) OSR protocols and precomputed embeddings for 4 datasets: IJB-B, IJB-C, Whale and VB-Eval. This data is sufficient to reproduce tables and figures from the paper.  
We also provide [checkpoints](https://drive.google.com/file/d/11zTq8NOJcxg3Bst6_KHXtJvPKCg9hgCx/view?usp=sharing) for pretrained sample quality estimators: SCF, PFE and ScaleFace. 
   - Extract the `datasets` and `model_weights` folders and place them in the root directory of the project. The structure should look like this:
     ```
     project_dir
     │   README.md
     └───datasets
     └───model_weights
     │
     │   ...
     ```

1. **Building the Docker Image**:
   - Navigate to the `docker_scripts` directory and run the build script:
     ```bash
     cd docker_scripts
     bash build.sh
     ```
   - The image building process takes approximately 10 minutes and will install all necessary dependencies.

2. **(Optional) Creating a Development Container**:
   - If you want to create a container for development, execute the following commands:
     ```bash
     cd docker_scripts
     bash launch_container.sh
     ```
   - After running the above commands, you can connect to the container using Visual Studio Code (VSCode).

## Method Evaluation
To create plots and tables for filtering test samples, follow these steps:  
1. **Running the comparison of uncertainty estimation methods**:  
   Navigate to the `scripts` directory and run the script to compute the main table (Table 1):
   ```bash
   cd scripts
   bash evaluate_filtering.sh
   ```
   To compare normalization strategies (Table 2) run `bash evaluate_filtering.sh`.  
2. **Storing Output**:  
   - All plots and tables will be saved in the `outputs/experiments/filter` directory.
   - For example, metrics of methods tested on the IJB-C datasets with a false positive identification rate (FPIR) of 0.1 will be saved in `outputs/experiments/filter/open_set_identification/IJBC/filter_plots/0.1`  
3. **Configuration**:  
   - The uncertainty estimation methods are described in configuration files `configs/uncertainty_benchmark/main_table.yaml` and `configs/uncertainty_benchmark/normalization_strategies.yaml`.  
4. **(Optional) Latex tables**  
   You can generate latex-formated table from raw .csv produces by main evaluation script with `bash create_latex_table.sh`

## Sample quality estimators (SCF, PFE, ScaleFace) training and inference
1. **Datasets**  
In order to train models you need to download datasets:  
- [MS1M-ArcFace](https://github.com/deepinsight/insightface/tree/master/recognition/_datasets_) is used to train sample quality estimators for face domaine.
Extract downloaded data in `datasets/ms1m`:
     ```
     datasets/ms1m
     │   train.rec
     │   train.idx
     │   ...
     ```
- [HappyWhale](https://github.com/knshnb/kaggle-happywhale-1st-place/blob/master/input/README.md). Please follow the instructions to download images for whale dataset. Resulting directory should have this structure:
     ```
     datasets/whale_images
     │   fullbody_test.csv
     │   fullbody_train.csv
     │    ...
     └───test_images
     └───train_images
     │   ...
     ```
 - [ArcFace](https://drive.google.com/file/d/1aC4zf2Bn0xCVH_ZtEuQipR2JvRb1bf8o/view) load images for IJB-B, IJB-C to compute embeddings. 
     ```
     datasets/arcface_ijb
     └───IJBB
     │   └───embeddings
     │   └───loose_crop
     │   └───meta
     └───IJBC
         └───embeddings
         └───loose_crop
         └───meta
     ```
 - [VoxBlink](https://github.com/VoxBlink2/ScriptsForVoxBlink2/tree/main) Please contact authors for VoxBlink dataset to request access. Extract using `utils/create_voxblink_embs.py` embeddings and bottleneck features of VoxBlink2 and VoxBlink-clean datasets in order to train and test models. 
2. **Train models**   
Train sample quaility model using script:
   ```bash
   cd scripts
   bash train_unc.sh
   ```
Select needed model and dataset using config option `-cn=scf_ms1m`, for example  
3. **Model inference**  
Use `predict_unc.sh` script to compute embeddings.