#!/bin/bash
#SBATCH --job-name=gradmaps    # Job name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=10          # CPU cores per task
#SBATCH --mem=100G                   # Memory per node
#SBATCH --gres=gpu:1               # Number of GPUs
#SBATCH --partition=gpu-a100

export CUDA_VISIBLE_DEVICES=0

MODEL_NAME=${1:?ERROR: pass model_name as first arg}
LAYER_NAME=${2:?ERROR: pass layer_name as second arg}
MODEL_PATH=${3:?ERROR: pass model_path as third arg}
ARCHITECTURE=${4:?ERROR: pass architecture as fourth arg}
# 5th arg (optional): recurrent timestep τ. 0 = first unroll step (no lateral yet),
# 9 = last step with T=10. Omit for the last step.
TIMESTEP=${5:-}
EXTRA=""
[[ -n "$TIMESTEP" ]] && EXTRA="--timestep $TIMESTEP"
python -m src.experiments.erf.compute_gradmaps \
    --model_name ${MODEL_NAME} \
    --layer_name ${LAYER_NAME} \
    --architecture ${ARCHITECTURE} \
    --model_path "${MODEL_PATH}" \
    ${EXTRA}