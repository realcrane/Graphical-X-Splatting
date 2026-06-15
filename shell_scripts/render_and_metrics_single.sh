#!/bin/bash
# Render + metrics for a SINGLE model output directory.
# Called by shell_scripts/slurm_render_and_metrics.sh (or manually).
# Run from the repo root.
#
# Usage: bash shell_scripts/render_and_metrics_single.sh <gpu_id> <model_path> [--force]

GPU_ID=${1:-0}
MODEL_PATH=${2:?Usage: render_and_metrics_single.sh <gpu_id> <model_path> [--force]}
FORCE_REPROCESS=false

for arg in "$@"; do
    if [ "$arg" = "--force" ]; then
        FORCE_REPROCESS=true
    fi
done

export CUDA_VISIBLE_DEVICES=$GPU_ID

# Remove trailing slash
MODEL_PATH="${MODEL_PATH%/}"

echo "=========================================="
echo "Rendering and metrics for: $MODEL_PATH"
echo "GPU ID: $GPU_ID"
echo "Force reprocess: $FORCE_REPROCESS"
echo "=========================================="

# Skip if already processed (unless --force)
if [ "$FORCE_REPROCESS" = false ] && [ -f "$MODEL_PATH/test_results.json" ]; then
    echo "Skipping $MODEL_PATH (test_results.json already exists, use --force to override)"
    exit 0
fi

CONFIG_PATH="$MODEL_PATH/cfg_args.json"

if [ ! -f "$CONFIG_PATH" ]; then
    echo "Warning: No cfg_args.json found in $MODEL_PATH. Skipping render but will try metrics."
else
    echo "Using config: $CONFIG_PATH"

    echo "Rendering iteration 30000..."
    python render.py -m "$MODEL_PATH" --config "$CONFIG_PATH" --render-videos --iteration 30000 --skip_train

    echo "Rendering _best_test..."
    python render.py -m "$MODEL_PATH" --config "$CONFIG_PATH" --render-videos --iteration _best_test --skip_train
fi

echo "Calculating metrics..."
python metrics.py -m "$MODEL_PATH" --skip_train

# Clean up PNGs
echo "Cleaning up PNG images..."
if [ -d "$MODEL_PATH/test" ]; then
    find "$MODEL_PATH/test" -type f -name "*.png" -delete
    echo "Removed PNG images from test folder"
fi
if [ -d "$MODEL_PATH/imgs" ]; then
    find "$MODEL_PATH/imgs" -type f -name "*.png" -delete
    echo "Removed PNG images from imgs folder"
fi

echo "Completed: $MODEL_PATH"
