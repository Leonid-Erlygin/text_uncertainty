# Gallery-Aware Uncertainty Estimation For Open-Set Face Recognition

### Project Configuration
System Requirements:
- Docker version 25.0.3
- CUDA Version: 12.2

To configure project dependecies please build your docker image using following commands:
```bash
cd docker_scripts
bash build.sh
```
image building process takes about 10 min, and it will install all the dependecies.

If you want to create development container please run following commands:
```bash
cd docker_scripts
bash launch_container.sh
```

### Method evaluation
In order to create rejection plots please run following commands:
```bash
cd scripts
bash evaluate_filtering.sh
```
