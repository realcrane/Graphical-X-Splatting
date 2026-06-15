#!/bin/bash

GPU_ID=${1:-0}
OUTPUT_DIR=${2:-output}
FORCE_REPROCESS=false

# Check for --force flag
for arg in "$@"; do
    if [ "$arg" = "--force" ]; then
        FORCE_REPROCESS=true
    fi
done

export CUDA_VISIBLE_DEVICES=$GPU_ID

echo "=========================================="
echo "Rendering and metrics calculation script"
echo "GPU ID: $GPU_ID"
echo "Output directory: $OUTPUT_DIR"
echo "Force reprocess: $FORCE_REPROCESS"
echo "=========================================="

# Find all result folders in the output directory
for MODEL_PATH in "$OUTPUT_DIR"/*/ ; do
    # Remove trailing slash
    MODEL_PATH=${MODEL_PATH%/}
    
    # Skip if not a directory
    if [ ! -d "$MODEL_PATH" ]; then
        continue
    fi
    
    # Skip if test_results.json already exists (unless --force flag is set)
    if [ "$FORCE_REPROCESS" = false ] && [ -f "$MODEL_PATH/test_results.json" ]; then
        echo "Skipping $MODEL_PATH (test_results.json already exists, use --force to override)"
        continue
    fi
    
    echo ""
    echo "=========================================="
    echo "Processing: $MODEL_PATH"
    echo "=========================================="
    
    # Use the cfg_args.json file from the model output folder
    CONFIG_PATH="$MODEL_PATH/cfg_args.json"
    
    if [ ! -f "$CONFIG_PATH" ]; then
        echo "Warning: No cfg_args.json found in $MODEL_PATH. Skipping render but will try metrics."
    else
        echo "Using config: $CONFIG_PATH"
        
        # Render iteration 30000
        echo "Rendering iteration 30000..."
        python render.py -m "$MODEL_PATH" --config "$CONFIG_PATH" --render-videos --iteration 30000 --skip_train
        
        # Render _best_test
        echo "Rendering _best_test..."
        python render.py -m "$MODEL_PATH" --config "$CONFIG_PATH" --render-videos --iteration _best_test --skip_train
    fi
    
    # Calculate metrics
    echo "Calculating metrics..."
    python metrics.py -m "$MODEL_PATH" --skip_train
    
    # Clean up PNG images after metrics calculation
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
done

echo ""
echo "=========================================="
echo "All folders processed!"
echo "=========================================="
