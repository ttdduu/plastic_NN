#!/bin/bash
#SBATCH --job-name=separability    # Job name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=10          # CPU cores per task
#SBATCH --mem=100G                   # Memory per node
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --partition=gpu-a100
export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=2

# Log GPU information
echo "Job starting, initial GPU status:"
nvidia-smi

python -m src.manifold_separability.mftma_analysis