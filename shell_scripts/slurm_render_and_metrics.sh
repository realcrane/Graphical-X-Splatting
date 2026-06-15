#!/bin/bash
# Submit render + metrics jobs in parallel via SLURM — one job per output scene.
#
# Usage:
#   ./shell_scripts/slurm_render_and_metrics.sh [OPTIONS]
#
# Options:
#   --output-dir DIR   Output directory (default: output)
#   --time HH:MM:SS    Wall-time per job  (default: 04:00:00)
#   --force             Re-process scenes that already have test_results.json
#   --delay OFFSET      Defer job start (e.g. 4hours, 30minutes, 1days). Passed to sbatch --begin=now+OFFSET
#   --dry-run           Print sbatch commands without submitting
#
# Examples:
#   ./shell_scripts/slurm_render_and_metrics.sh
#   ./shell_scripts/slurm_render_and_metrics.sh --force --time 06:00:00
#   ./shell_scripts/slurm_render_and_metrics.sh --delay 4hours
#   ./shell_scripts/slurm_render_and_metrics.sh --dry-run

OUTPUT_DIR="output"
TIME="04:00:00"
FORCE=""
DELAY=""
DRY_RUN=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
        --time)       TIME="$2";       shift 2 ;;
        --force)      FORCE="--force"; shift ;;
        --delay)      DELAY="$2";      shift 2 ;;
        --dry-run)    DRY_RUN=true;    shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

COUNT=0

for MODEL_PATH in "$OUTPUT_DIR"/*/; do
    # Remove trailing slash
    MODEL_PATH="${MODEL_PATH%/}"

    # Skip non-directories
    [ ! -d "$MODEL_PATH" ] && continue

    # Skip already-processed scenes (unless --force)
    if [ -z "$FORCE" ] && [ -f "$MODEL_PATH/test_results.json" ]; then
        echo "SKIP  $MODEL_PATH (already has results, use --force to override)"
        continue
    fi

    SCENE_NAME=$(basename "$MODEL_PATH")
    JOB_NAME="rm_${SCENE_NAME}"

    BEGIN_FLAG=""
    if [ -n "$DELAY" ]; then
        BEGIN_FLAG="--begin=now+${DELAY}"
    fi

    SBATCH_CMD="sbatch \
  --job-name=${JOB_NAME} \
  --gres=gpu:1 \
  --time=${TIME} \
  ${BEGIN_FLAG} \
  --output=logs/%x_%j.out \
  --error=logs/%x_%j.err \
  --wrap=\"bash shell_scripts/render_and_metrics_single.sh 0 ${MODEL_PATH} ${FORCE}\""

    if [ "$DRY_RUN" = true ]; then
        echo "[DRY RUN] $SBATCH_CMD"
    else
        echo "Submitting: $SCENE_NAME"
        eval "$SBATCH_CMD"
    fi

    COUNT=$((COUNT + 1))
done

echo ""
echo "Submitted $COUNT render+metrics jobs."
