# Gallery-Aware Uncertainty Estimation For Open-Set Face Recognition

### Project Configuration
System Requirements:
- Docker version 25.0.3
- CUDA Version: 12.2

Repository preparation steps:
1. Download datasets via link https://disk.yandex.ru/d/dtX-48519lKt-A
2. Extract `datasets` folder and place it inside project directory at root level:
```
face_ue
│   README.md
│      
└───datasets
│
│   ...
```

To configure project dependecies please build your docker image using following commands:
```bash
cd docker_scripts
bash build.sh
```
image building process takes about 10 min, and it will install all the dependecies.

(optional) If you want to create development container please run following commands:
```bash
cd docker_scripts
bash launch_container.sh
```
afterwards you could attach to container using vscode

### Method evaluation
In order to create rejection plots please run following commands:
```bash
cd scripts
bash evaluate_filtering.sh
```
first run might take a while...  
All the plots will be stored in directory `outputs/experiments/filtering_plots`  
In this example plots from the paper will be stored at `outputs/experiments/filtering_plots/open_set_identification/IJBC/filter_plots/0.05`  
because we test our method on dataset IJBC at FAR $0.05$, which is specified in config

Evaluation script will be run with config `configs/uncertainty_benchmark/evaluate_filtering.yaml`  