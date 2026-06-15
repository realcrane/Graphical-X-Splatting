#!/bin/bash
# End-to-end 4D init for all Google Immersive scenes, on GPU 0:
#   Stage 1 — undistorted frames + per-frame COLMAP poses (colmap_from_video_immersive.py)
#   Stage 2 — RoMA per-frame point clouds + merge into points4D.ply
# Stage 1 reads data/google_immersive/<scene>/ (videos + models.json) and writes the
# sibling data/google_immersive/<scene>_dist/ that Stage 2 and training consume.

set -e
export CUDA_VISIBLE_DEVICES=0

SCENES=(01_Welder 02_Flames 05_Horse 06_Goats 10_Alexa)
STARTFRAME=0
ENDFRAME=30

for scene in "${SCENES[@]}"; do
    echo "=== [$scene] Stage 1: undistorted frames + per-frame COLMAP poses ==="
    python3 preprocessing/colmap_from_video_immersive.py \
        --videopath data/google_immersive/$scene \
        --startframe $STARTFRAME --endframe $ENDFRAME

    echo "=== [$scene] Stage 2: RoMA per-frame point clouds ==="
    python3 preprocessing/pcd_from_images_immersive.py \
        --base_dir   data/google_immersive/${scene}_dist \
        --output_dir data/google_immersive/${scene}_dist/point_clouds

    echo "=== [$scene] Stage 2: merge into points4D.ply ==="
    python3 preprocessing/pcd_merge_4d.py \
        --input_dir data/google_immersive/${scene}_dist/point_clouds \
        --output    data/google_immersive/${scene}_dist/points4D.ply
done
