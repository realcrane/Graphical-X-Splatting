#!/bin/bash
# Train several scenes of one dataset with a single config (sequentially).
# Dataset-agnostic: select the dataset root with --data-root (default: data/N3DV).
# Run from the repo root.

# Directory of this script, so it can call its sibling regardless of the caller's CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DATA_ROOT="data/N3DV"
DRY_RUN=false

# Optional flags may appear (in any order) before the positional args.
while [ "$#" -gt 0 ] && [ "${1#--}" != "$1" ]; do
    case "$1" in
        --dry-run)   DRY_RUN=true; shift ;;
        --data-root) DATA_ROOT="${2%/}"; shift 2 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 [--dry-run] [--data-root DIR] <gpu_id> <config_path> <scene1> [scene2] ..."
    echo "  --data-root DIR   Dataset root containing the scene folders (default: data/N3DV)"
    echo ""
    echo "Example (N3DV):      $0 0 configs/graphits_standard_n3dv.json cook_spinach coffee_martini"
    echo "Example (Immersive): $0 --data-root data/google_immersive 0 configs/graphits_standard_gi.json 01_Welder_dist 02_Flames_dist"
    exit 1
fi

GPU_ID=$1
CONFIG_PATH=$2
shift 2

CONFIG_NAME=$(basename "$CONFIG_PATH" .json)

if [ "$DRY_RUN" = true ]; then
    echo "DRY RUN MODE - No commands will be executed"
    echo ""
fi

for SCENE in "$@"; do
    SOURCE_PATH="$DATA_ROOT/$SCENE"
    MODEL_PATH="output/${SCENE}_${CONFIG_NAME}"

    echo "=========================================="
    echo "Running experiment for scene: $SCENE"
    echo "Source path: $SOURCE_PATH"
    echo "Model path: $MODEL_PATH"
    echo "=========================================="

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] Would execute: $SCRIPT_DIR/run_experiment.sh $GPU_ID $SOURCE_PATH $MODEL_PATH $CONFIG_PATH"
    else
        "$SCRIPT_DIR/run_experiment.sh" "$GPU_ID" "$SOURCE_PATH" "$MODEL_PATH" "$CONFIG_PATH"
    fi

    echo "Completed experiment for scene: $SCENE"
    echo ""
done

echo "All experiments completed!"
