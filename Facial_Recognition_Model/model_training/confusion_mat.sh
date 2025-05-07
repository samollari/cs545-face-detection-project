#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=1                # One task per GPU
#SBATCH --gres=gpu:1              # Request 2 GPUs
#SBATCH -n 8
#SBATCH --mem=16g
#SBATCH -J "confusion_mat"
#SBATCH -o confusion_mat.out
#SBATCH -e confusion_mat.err
#SBATCH -p academic
#SBATCH -t 2:00:00

# module load miniconda3
# module load cuda/11.8.0/4w5kyjs 

# Uncomment module lines above if using this as SLURM request script

set -e

ENV_NAME="face-recognition"
PYTHON_SCRIPT="confusion_matrix.py"

source "$CONDA_PREFIX/etc/profile.d/conda.sh"

conda activate "$ENV_NAME"

echo "Using Python at: $(which python)"
python -c "import torch; print('Torch version:', torch.__version__)"

# Launch with torchrun for DDP
echo "Scenario 1"
python "$PYTHON_SCRIPT" --weights ./../trained_models/80_acc.pth --data_zip ./../datasets/lfw.zip --img_dir lfw-deepfunneled/lfw-deepfunneled --output ./../real_mat.png --num_classes 10572
echo "Scenario 2"
python "$PYTHON_SCRIPT" --weights ./../trained_models/73_raw.pth --data_zip ./../datasets/lfw.zip --img_dir lfw-deepfunneled/lfw-deepfunneled --output ./../raw_mat.png
echo "Scenario 3"
python "$PYTHON_SCRIPT" --weights ./../trained_models/68_processed.pth --data_zip ./../datasets/lfw.zip --img_dir lfw-deepfunneled/lfw-deepfunneled --output ./../processed_mat.png
echo "Scenario 4"
python "$PYTHON_SCRIPT" --weights ./../trained_models/63_edges.pth --data_zip ./../datasets/lfw_edges.zip --img_dir lfw_edges --output ./../edges_mat.png