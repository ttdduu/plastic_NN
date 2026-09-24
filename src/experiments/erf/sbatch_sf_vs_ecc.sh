#!/bin/bash
#SBATCH --job-name=lay1   # Job name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=10          # CPU cores per task
#SBATCH --mem=100G                   # Memory per node
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --partition=gpu

export PYTHONUNBUFFERED=1
python -u -m src.experiments.erf.netStats