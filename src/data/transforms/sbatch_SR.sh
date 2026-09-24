#!/bin/bash
#SBATCH --job-name=gogogo    # Job name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=10          # CPU cores per task
#SBATCH --mem=100G                   # Memory per node
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --partition=gpu
export CUDA_LAUNCH_BLOCKING=1
python -m src.data.data_preparation_scripts.create_filtered_dataset
echo "Job finished"