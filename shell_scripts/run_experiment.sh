#!/bin/bash
# Train a single scene (dataset-agnostic — works for any -s source path).
# Run from the repo root. Usually invoked by run_experiments.sh.

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <gpu_id> <source_path> <model_path> <config_path>"
    echo "Example (N3DV):      $0 0 data/N3DV/cook_spinach output/cook_spinach_std configs/graphits_standard_n3dv.json"
    echo "Example (Immersive): $0 0 data/google_immersive/01_Welder_dist output/welder_std configs/graphits_standard_gi.json"
    exit 1
fi

GPU_ID=$1
SOURCE_PATH=$2
MODEL_PATH=$3
CONFIG_PATH=$4

export CUDA_VISIBLE_DEVICES=$GPU_ID

python train.py -s $SOURCE_PATH -m $MODEL_PATH --config $CONFIG_PATH
