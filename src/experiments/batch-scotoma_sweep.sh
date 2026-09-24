#!/bin/bash
#SBATCH --job-name=gogogo    # Job name
#SBATCH --nodes=1                   # Number of nodes
#SBATCH --ntasks=1                  # Number of tasks
#SBATCH --cpus-per-task=10          # CPU cores per task
#SBATCH --mem=200G                   # Memory per node
#SBATCH --gres=gpu:1               # Number of GPUs
export CUDA_LAUNCH_BLOCKING=1
# export CUDA_VISIBLE_DEVICES=2

# Log GPU information
echo "Job starting, initial GPU status:"
nvidia-smi

# Wandb settings
export WANDB_MODE=offline           # Run in offline mode
export WANDB_ENTITY=ttdduu-cerco
export WANDB_PROJECT=scotoma_exp
export WANDB_TIMEOUT=10
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#ADDITIONAL_ARGS="$@"
echo "before calling the script"
# Positional args: $1=scotoma_radius, $2=model_architecture, $3=conv_weights, $4=lc_weights, $5=weights_path
# Pass "None" to skip an optional arg. Example:
#   sbatch batch-scotoma_sweep.sh 0 cornet_dwsep_lc None None /path/to/checkpoint.pth
SCOTOMA_RADIUS=${1:-0}
MODEL_ARCHITECTURE=${2:-}
CONV_WEIGHTS=${3:-}
MODEL_LC_WEIGHTS=${4:-}
MODEL_WEIGHTS_PATH=${5:-}
DATASET=${6:-}
SPATIAL_LOSS_ALPHA=${7:-}
EXTRA_ARGS=""
[[ -n "$MODEL_ARCHITECTURE" && "$MODEL_ARCHITECTURE" != "None" ]] && EXTRA_ARGS="$EXTRA_ARGS --model-architecture $MODEL_ARCHITECTURE"
[[ -n "$CONV_WEIGHTS"       && "$CONV_WEIGHTS"       != "None" ]] && EXTRA_ARGS="$EXTRA_ARGS --model-conv_weights $CONV_WEIGHTS"
[[ -n "$MODEL_LC_WEIGHTS"   && "$MODEL_LC_WEIGHTS"   != "None" ]] && EXTRA_ARGS="$EXTRA_ARGS --model-lc_weights $MODEL_LC_WEIGHTS"
[[ -n "$MODEL_WEIGHTS_PATH" && "$MODEL_WEIGHTS_PATH" != "None" ]] && EXTRA_ARGS="$EXTRA_ARGS --model-weights_path $MODEL_WEIGHTS_PATH"
[[ -n "$DATASET" && "$DATASET" != "None" ]] && EXTRA_ARGS="$EXTRA_ARGS --data-dataset $DATASET"
[[ -n "$SPATIAL_LOSS_ALPHA" && "$SPATIAL_LOSS_ALPHA" != "None" ]] && EXTRA_ARGS="$EXTRA_ARGS --training-spatial_loss_alpha $SPATIAL_LOSS_ALPHA"
python -m src.experiments.scotoma_visualization.scotoma_parameter_sweep \
    --data-scotoma_radius ${SCOTOMA_RADIUS} \
    ${EXTRA_ARGS}
#python -m src.simpleScript.py

echo "Job finished"
