#!/bin/bash
#SBATCH --job-name=gmp_fit    # Job name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=10          # CPU cores per task
#SBATCH --mem=100G                   # Memory per node
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --partition=gpu-a100

export CUDA_VISIBLE_DEVICES=0

MODEL_NAME=${1:?ERROR: pass model_name as first arg, e.g. sbatch gradmap_analysis_sbatch.sh mqi6f48u stages.0.0.dwconv}
LAYER_NAME=${2:?ERROR: pass layer_name as second arg, e.g. sbatch gradmap_analysis_sbatch.sh mqi6f48u stages.0.0.dwconv}
# 3rd arg (optional): recurrent timestep τ the gradmaps were computed at (must
# match the _t<τ> suffix compute_gradmaps wrote, e.g. 9). Omit for legacy gradmaps.
TIMESTEP=${3:-}
EXTRA=""
[[ -n "$TIMESTEP" ]] && EXTRA="--timestep $TIMESTEP"
python -m src.experiments.erf.gradmap_analysis --model_name ${MODEL_NAME} --layer_name ${LAYER_NAME} ${EXTRA}