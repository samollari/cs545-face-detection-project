#!/bin/bash
#SBATCH -N 1
#SBATCH --ntasks=2                # One task per GPU
#SBATCH --gres=gpu:2              # Request 2 GPUs
#SBATCH -n 8
#SBATCH --mem=16g
#SBATCH -J "Train_ResNetIRSE"
#SBATCH -o Train_ResNetIRSE.out
#SBATCH -e Train_ResNetIRSE.err
#SBATCH -p academic
#SBATCH -t 8:00:00

# module load miniconda3
# module load cuda/11.8.0/4w5kyjs

set -e

ENV_NAME="face-recognition"
PYTHON_SCRIPT="train.py"

source "$CONDA_PREFIX/etc/profile.d/conda.sh"

conda activate "$ENV_NAME"

echo "Using Python at: $(which python)"
python -c "import torch; print('Torch version:', torch.__version__)"

# Launch with torchrun for DDP
torchrun --nproc_per_node=2 "$PYTHON_SCRIPT"