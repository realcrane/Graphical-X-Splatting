#!/bin/bash
# End-to-end 4D init for all N3DV scenes, on GPU 0:
#   Stage 1 — frames + COLMAP poses (colmap_from_video_n3dv.py)
#   Stage 2 — RoMA per-frame point clouds + merge into points4D.ply
# Assumes each scene folder already contains the per-camera camXX.mp4 videos.

set -e
export CUDA_VISIBLE_DEVICES=0

SCENES=(cook_spinach coffee_martini cut_roasted_beef flame_salmon flame_steak sear_steak)

for scene in "${SCENES[@]}"; do
    echo "=== [$scene] Stage 1: frames + COLMAP poses ==="
    python3 preprocessing/colmap_from_video_n3dv.py data/N3DV/$scene/

    echo "=== [$scene] Stage 2: RoMA per-frame point clouds ==="
    python3 preprocessing/pcd_from_images_n3dv.py \
        --colmap_dir data/N3DV/$scene/ \
        --image_dir  data/N3DV/$scene/images/ \
        --output_dir data/N3DV/$scene/point_clouds/

    echo "=== [$scene] Stage 2: merge into points4D.ply ==="
    python3 preprocessing/pcd_merge_4d.py \
        --input_dir data/N3DV/$scene/point_clouds \
        --output    data/N3DV/$scene/points4D.ply
done
