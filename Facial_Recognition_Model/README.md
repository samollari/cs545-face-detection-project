# cs545-face-detection-project

## Model Training

### Set Up

- [ ] Install Conda (or Miniconda)
- [ ] Clone this repo
- [ ] Create conda environment with the following:
```
conda env create -f environment.yml
```
This should create a conda environment called face-recognition which contains all of the appropriate packages for running the training and evaluation loop

In addition to creating the conda environment, all datasets need to be downloaded and placed in the `datasets` directory. All datasets used for this project can be found here: https://drive.google.com/drive/u/1/folders/11nIBQhPW4oChXua9knfDDgvT2LqEIwHr

#### Note: Do not unzip the zipped dataset folders after downloading them from the drive link. Unzipping these folders will cause the training pipeline to break.  

### Running
The training script for this project was originally ran on the ARC GPU Cluster through a SLURM job request found in `start_train.sh`. Using the shell script will launch the script using Pytorch's distributed data parallel processing for multiple gpus. If running the script on one gpu, it is recommended to launch the script in the shell with the command below:
```
python train.py
```

It should be noted that running this script in the shell may break on the first run due to parameters not properly set inside the `train.py` file. In order to edit run configurations quickly, open the train.py file and edit the parameter block shown below in the python file. `training_scenario` controls which dataset to use for training the model.

```
## Switches and Variable Params
# 1: Real Data
# 2: Unprocessed Synthetic
# 3: Contrast+Histogram Processed Synthetic
# 4: Edgemap Processed Synthetic

training_scenario = 1 
use_distributed = True # Enable this if using multiple gpus
b_size = 128 # batch size
workers = 4 # Number of workers
num_gpu=2 # Number of GPUs
```

While training, the best performing models will be saved to the `trained_models` directory. If you would like to reproduce the confusion matrices for each model in the paper, download them from the drive link and place them in the `trained_models` directory. Afterwards, run `confusion_mat.sh` in the `base` conda environment. This shell script is also based off of a SLURM job request and has been adapted for local use.
