#!/bin/bash

set -e

# ---------------GRAPHITS STANDARD (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphits_standard --config configs/graphits_standard_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphits_standard --config configs/graphits_standard_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphits_standard --config configs/graphits_standard_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphits_standard --config configs/graphits_standard_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphits_standard --config configs/graphits_standard_gi.json


# ---------------GRAPHIGS STANDARD (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphigs_standard --config configs/graphigs_standard_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphigs_standard --config configs/graphigs_standard_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphigs_standard --config configs/graphigs_standard_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphigs_standard --config configs/graphigs_standard_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphigs_standard --config configs/graphigs_standard_gi.json


# ---------------GRAPHITS OMIT_10PCT_CAMS (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphits_omit_10pct_cams --config configs/graphits_sparse_views_10_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphits_omit_10pct_cams --config configs/graphits_sparse_views_10_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphits_omit_10pct_cams --config configs/graphits_sparse_views_10_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphits_omit_10pct_cams --config configs/graphits_sparse_views_10_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphits_omit_10pct_cams --config configs/graphits_sparse_views_10_gi.json


# ---------------GRAPHIGS OMIT_10PCT_CAMS (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphigs_omit_10pct_cams --config configs/graphigs_sparse_views_10_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphigs_omit_10pct_cams --config configs/graphigs_sparse_views_10_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphigs_omit_10pct_cams --config configs/graphigs_sparse_views_10_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphigs_omit_10pct_cams --config configs/graphigs_sparse_views_10_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphigs_omit_10pct_cams --config configs/graphigs_sparse_views_10_gi.json


# ---------------GRAPHITS OMIT_30PCT_CAMS (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphits_omit_30pct_cams --config configs/graphits_sparse_views_30_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphits_omit_30pct_cams --config configs/graphits_sparse_views_30_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphits_omit_30pct_cams --config configs/graphits_sparse_views_30_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphits_omit_30pct_cams --config configs/graphits_sparse_views_30_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphits_omit_30pct_cams --config configs/graphits_sparse_views_30_gi.json


# ---------------GRAPHIGS OMIT_30PCT_CAMS (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphigs_omit_30pct_cams --config configs/graphigs_sparse_views_30_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphigs_omit_30pct_cams --config configs/graphigs_sparse_views_30_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphigs_omit_30pct_cams --config configs/graphigs_sparse_views_30_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphigs_omit_30pct_cams --config configs/graphigs_sparse_views_30_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphigs_omit_30pct_cams --config configs/graphigs_sparse_views_30_gi.json


# ---------------GRAPHITS OMIT_50PCT_CAMS (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphits_omit_50pct_cams --config configs/graphits_sparse_views_50_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphits_omit_50pct_cams --config configs/graphits_sparse_views_50_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphits_omit_50pct_cams --config configs/graphits_sparse_views_50_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphits_omit_50pct_cams --config configs/graphits_sparse_views_50_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphits_omit_50pct_cams --config configs/graphits_sparse_views_50_gi.json


# ---------------GRAPHIGS OMIT_50PCT_CAMS (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphigs_omit_50pct_cams --config configs/graphigs_sparse_views_50_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphigs_omit_50pct_cams --config configs/graphigs_sparse_views_50_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphigs_omit_50pct_cams --config configs/graphigs_sparse_views_50_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphigs_omit_50pct_cams --config configs/graphigs_sparse_views_50_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphigs_omit_50pct_cams --config configs/graphigs_sparse_views_50_gi.json


# ---------------GRAPHITS FPS_10 (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphits_fps_10 --config configs/graphits_sparse_frames_10fps_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphits_fps_10 --config configs/graphits_sparse_frames_10fps_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphits_fps_10 --config configs/graphits_sparse_frames_10fps_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphits_fps_10 --config configs/graphits_sparse_frames_10fps_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphits_fps_10 --config configs/graphits_sparse_frames_10fps_gi.json


# ---------------GRAPHIGS FPS_10 (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphigs_fps_10 --config configs/graphigs_sparse_frames_10fps_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphigs_fps_10 --config configs/graphigs_sparse_frames_10fps_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphigs_fps_10 --config configs/graphigs_sparse_frames_10fps_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphigs_fps_10 --config configs/graphigs_sparse_frames_10fps_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphigs_fps_10 --config configs/graphigs_sparse_frames_10fps_gi.json


# ---------------GRAPHITS FPS_20 (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphits_fps_20 --config configs/graphits_sparse_frames_20fps_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphits_fps_20 --config configs/graphits_sparse_frames_20fps_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphits_fps_20 --config configs/graphits_sparse_frames_20fps_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphits_fps_20 --config configs/graphits_sparse_frames_20fps_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphits_fps_20 --config configs/graphits_sparse_frames_20fps_gi.json


# ---------------GRAPHIGS FPS_20 (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphigs_fps_20 --config configs/graphigs_sparse_frames_20fps_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphigs_fps_20 --config configs/graphigs_sparse_frames_20fps_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphigs_fps_20 --config configs/graphigs_sparse_frames_20fps_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphigs_fps_20 --config configs/graphigs_sparse_frames_20fps_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphigs_fps_20 --config configs/graphigs_sparse_frames_20fps_gi.json


# ---------------GRAPHITS NOSYNC_50_FPS_20 (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphits_nosync_50_fps_20 --config configs/graphits_unsync_50_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphits_nosync_50_fps_20 --config configs/graphits_unsync_50_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphits_nosync_50_fps_20 --config configs/graphits_unsync_50_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphits_nosync_50_fps_20 --config configs/graphits_unsync_50_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphits_nosync_50_fps_20 --config configs/graphits_unsync_50_gi.json


# ---------------GRAPHIGS NOSYNC_50_FPS_20 (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphigs_nosync_50_fps_20 --config configs/graphigs_unsync_50_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphigs_nosync_50_fps_20 --config configs/graphigs_unsync_50_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphigs_nosync_50_fps_20 --config configs/graphigs_unsync_50_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphigs_nosync_50_fps_20 --config configs/graphigs_unsync_50_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphigs_nosync_50_fps_20 --config configs/graphigs_unsync_50_gi.json


# ---------------GRAPHITS NOSYNC_90_FPS_20 (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphits_nosync_90_fps_20 --config configs/graphits_unsync_10_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphits_nosync_90_fps_20 --config configs/graphits_unsync_10_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphits_nosync_90_fps_20 --config configs/graphits_unsync_10_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphits_nosync_90_fps_20 --config configs/graphits_unsync_10_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphits_nosync_90_fps_20 --config configs/graphits_unsync_10_gi.json


# ---------------GRAPHIGS NOSYNC_90_FPS_20 (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphigs_nosync_90_fps_20 --config configs/graphigs_unsync_10_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphigs_nosync_90_fps_20 --config configs/graphigs_unsync_10_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphigs_nosync_90_fps_20 --config configs/graphigs_unsync_10_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphigs_nosync_90_fps_20 --config configs/graphigs_unsync_10_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphigs_nosync_90_fps_20 --config configs/graphigs_unsync_10_gi.json


# ---------------GRAPHITS FAULTY_CAMS (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphits_faulty_cams --config configs/graphits_faulty_cams_1_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphits_faulty_cams --config configs/graphits_faulty_cams_1_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphits_faulty_cams --config configs/graphits_faulty_cams_1_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphits_faulty_cams --config configs/graphits_faulty_cams_1_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphits_faulty_cams --config configs/graphits_faulty_cams_1_gi.json


# ---------------GRAPHIGS FAULTY_CAMS (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphigs_faulty_cams --config configs/graphigs_faulty_cams_1_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphigs_faulty_cams --config configs/graphigs_faulty_cams_1_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphigs_faulty_cams --config configs/graphigs_faulty_cams_1_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphigs_faulty_cams --config configs/graphigs_faulty_cams_1_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphigs_faulty_cams --config configs/graphigs_faulty_cams_1_gi.json


# ---------------GRAPHITS FAULTY_CAMS_2 (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphits_faulty_cams_2 --config configs/graphits_faulty_cams_2_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphits_faulty_cams_2 --config configs/graphits_faulty_cams_2_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphits_faulty_cams_2 --config configs/graphits_faulty_cams_2_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphits_faulty_cams_2 --config configs/graphits_faulty_cams_2_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphits_faulty_cams_2 --config configs/graphits_faulty_cams_2_gi.json


# ---------------GRAPHIGS FAULTY_CAMS_2 (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphigs_faulty_cams_2 --config configs/graphigs_faulty_cams_2_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphigs_faulty_cams_2 --config configs/graphigs_faulty_cams_2_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphigs_faulty_cams_2 --config configs/graphigs_faulty_cams_2_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphigs_faulty_cams_2 --config configs/graphigs_faulty_cams_2_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphigs_faulty_cams_2 --config configs/graphigs_faulty_cams_2_gi.json


# ---------------GRAPHITS FAULTY_CAMS_3 (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphits_faulty_cams_3 --config configs/graphits_faulty_cams_3_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphits_faulty_cams_3 --config configs/graphits_faulty_cams_3_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphits_faulty_cams_3 --config configs/graphits_faulty_cams_3_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphits_faulty_cams_3 --config configs/graphits_faulty_cams_3_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphits_faulty_cams_3 --config configs/graphits_faulty_cams_3_gi.json


# ---------------GRAPHIGS FAULTY_CAMS_3 (Google Immersive)---------------

CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/01_Welder_dist -m output/welder_graphigs_faulty_cams_3 --config configs/graphigs_faulty_cams_3_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/02_Flames_dist -m output/flames_graphigs_faulty_cams_3 --config configs/graphigs_faulty_cams_3_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/05_Horse_dist -m output/horse_graphigs_faulty_cams_3 --config configs/graphigs_faulty_cams_3_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/06_Goats_dist -m output/goats_graphigs_faulty_cams_3 --config configs/graphigs_faulty_cams_3_gi.json
CUDA_VISIBLE_DEVICES=0 python train.py -s data/google_immersive/10_Alexa_dist -m output/alexa_graphigs_faulty_cams_3 --config configs/graphigs_faulty_cams_3_gi.json
