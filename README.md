# Holistic Uncertainty Estimation For Open-Set Recognition

<!-- ### System Requirements
- **Docker Version**: 25.0.3
- **CUDA Version**: 12.2 -->

### Repository Setup Steps

1. **Downloading Datasets**:
   - Use the provided link to download the datasets: [Yandex Disk Link](https://disk.yandex.ru/d/TjOwePopUJmbpA)
   - Extract the `datasets` folder and place it in the root directory of the project. The structure should look like this:
     ```
     face_ue
     │   README.md
     │
     └───datasets
     │
     │   ...
     ```

2. **Building the Docker Image**:
   - Navigate to the `docker_scripts` directory and run the build script:
     ```bash
     cd docker_scripts
     bash build.sh
     ```
   - The image building process takes approximately 10 minutes and will install all necessary dependencies.

3. **(Optional) Creating a Development Container**:
   - If you want to create a container for development, execute the following commands:
     ```bash
     cd docker_scripts
     bash launch_container.sh
     ```
   - After running the above commands, you can connect to the container using Visual Studio Code (VSCode).

### Method Evaluation

To create plots for filtering test samples, follow these steps:

1. **Running the Uncertainty Estimation and Quality Metrics Script**:
   - Navigate to the `scripts` directory and run the uncertainty estimation script on the IJBB and IJBC datasets:
     ```bash
     cd scripts
     bash evaluate_filtering.sh
     ```
   - Navigate to the `scripts` directory and run the uncertainty estimation script on the Whale dataset:
     ```bash
     cd scripts
     bash evaluate_filtering_whale.sh
     ```
   - The first run may take some time.

2. **Storing Output**:
   - All plots will be saved in the `outputs/experiments/filtering_plots` directory.
   - For example, plots for the IJBC dataset will be saved in `outputs/experiments/filtering_plots/open_set_identification/IJBC/filter_plots/0.1` because the method is tested on the IJBC datasets with a False Acceptance Rate (FAR) of 0.1, as specified in the configuration.

3. **Configuration**:
   - The uncertainty estimation and quality metrics scripts will run with the configuration files `configs/uncertainty_benchmark/evaluate_filtering.yaml` and `configs/uncertainty_benchmark/evaluate_filtering_whale.yaml`.

### Additional Notes

- Ensure that Docker and CUDA are correctly installed and configured on your system.
- Ensure you have sufficient disk space and memory to process the dataset and Docker image.