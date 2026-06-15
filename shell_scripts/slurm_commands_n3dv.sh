#!/bin/bash
# Reference list of ready-to-paste sbatch commands for all N3DV ablation experiments.
# NOT intended to be executed as-is (that would submit EVERY job at once) — copy/paste
# the individual sbatch lines you need. Section headers below are comments.

# ---------------GRAPHITS STANDARD (N3DV)---------------

sbatch   --job-name=coffee_martini_graphits_standard   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphits_standard --config configs/graphits_standard_n3dv.json"

sbatch   --job-name=cook_spinach_graphits_standard   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphits_standard --config configs/graphits_standard_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphits_standard   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphits_standard --config configs/graphits_standard_n3dv.json"

sbatch   --job-name=flame_salmon_graphits_standard   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphits_standard --config configs/graphits_standard_n3dv.json"

sbatch   --job-name=flame_steak_graphits_standard   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphits_standard --config configs/graphits_standard_n3dv.json"

sbatch   --job-name=sear_steak_graphits_standard   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphits_standard --config configs/graphits_standard_n3dv.json"


# ---------------GRAPHIGS STANDARD (N3DV)---------------

sbatch   --job-name=coffee_martini_graphigs_standard   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphigs_standard --config configs/graphigs_standard_n3dv.json"

sbatch   --job-name=cook_spinach_graphigs_standard   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphigs_standard --config configs/graphigs_standard_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphigs_standard   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphigs_standard --config configs/graphigs_standard_n3dv.json"

sbatch   --job-name=flame_salmon_graphigs_standard   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphigs_standard --config configs/graphigs_standard_n3dv.json"

sbatch   --job-name=flame_steak_graphigs_standard   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphigs_standard --config configs/graphigs_standard_n3dv.json"

sbatch   --job-name=sear_steak_graphigs_standard   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphigs_standard --config configs/graphigs_standard_n3dv.json"


# ---------------GRAPHITS SPARSE_50 (N3DV)---------------

sbatch   --job-name=coffee_martini_graphits_sparse_50   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphits_sparse_50 --config configs/graphits_sparse_views_50_n3dv.json"

sbatch   --job-name=cook_spinach_graphits_sparse_50   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphits_sparse_50 --config configs/graphits_sparse_views_50_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphits_sparse_50   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphits_sparse_50 --config configs/graphits_sparse_views_50_n3dv.json"

sbatch   --job-name=flame_salmon_graphits_sparse_50   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphits_sparse_50 --config configs/graphits_sparse_views_50_n3dv.json"

sbatch   --job-name=flame_steak_graphits_sparse_50   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphits_sparse_50 --config configs/graphits_sparse_views_50_n3dv.json"

sbatch   --job-name=sear_steak_graphits_sparse_50   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphits_sparse_50 --config configs/graphits_sparse_views_50_n3dv.json"


# ---------------GRAPHIGS SPARSE_50 (N3DV)---------------

sbatch   --job-name=coffee_martini_graphigs_sparse_50   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphigs_sparse_50 --config configs/graphigs_sparse_views_50_n3dv.json"

sbatch   --job-name=cook_spinach_graphigs_sparse_50   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphigs_sparse_50 --config configs/graphigs_sparse_views_50_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphigs_sparse_50   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphigs_sparse_50 --config configs/graphigs_sparse_views_50_n3dv.json"

sbatch   --job-name=flame_salmon_graphigs_sparse_50   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphigs_sparse_50 --config configs/graphigs_sparse_views_50_n3dv.json"

sbatch   --job-name=flame_steak_graphigs_sparse_50   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphigs_sparse_50 --config configs/graphigs_sparse_views_50_n3dv.json"

sbatch   --job-name=sear_steak_graphigs_sparse_50   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphigs_sparse_50 --config configs/graphigs_sparse_views_50_n3dv.json"


# ---------------GRAPHITS SPARSE_70 (N3DV)---------------

sbatch   --job-name=coffee_martini_graphits_sparse_70   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphits_sparse_70 --config configs/graphits_sparse_views_30_n3dv.json"

sbatch   --job-name=cook_spinach_graphits_sparse_70   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphits_sparse_70 --config configs/graphits_sparse_views_30_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphits_sparse_70   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphits_sparse_70 --config configs/graphits_sparse_views_30_n3dv.json"

sbatch   --job-name=flame_salmon_graphits_sparse_70   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphits_sparse_70 --config configs/graphits_sparse_views_30_n3dv.json"

sbatch   --job-name=flame_steak_graphits_sparse_70   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphits_sparse_70 --config configs/graphits_sparse_views_30_n3dv.json"

sbatch   --job-name=sear_steak_graphits_sparse_70   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphits_sparse_70 --config configs/graphits_sparse_views_30_n3dv.json"


# ---------------GRAPHIGS SPARSE_70 (N3DV)---------------

sbatch   --job-name=coffee_martini_graphigs_sparse_70   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphigs_sparse_70 --config configs/graphigs_sparse_views_30_n3dv.json"

sbatch   --job-name=cook_spinach_graphigs_sparse_70   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphigs_sparse_70 --config configs/graphigs_sparse_views_30_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphigs_sparse_70   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphigs_sparse_70 --config configs/graphigs_sparse_views_30_n3dv.json"

sbatch   --job-name=flame_salmon_graphigs_sparse_70   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphigs_sparse_70 --config configs/graphigs_sparse_views_30_n3dv.json"

sbatch   --job-name=flame_steak_graphigs_sparse_70   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphigs_sparse_70 --config configs/graphigs_sparse_views_30_n3dv.json"

sbatch   --job-name=sear_steak_graphigs_sparse_70   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphigs_sparse_70 --config configs/graphigs_sparse_views_30_n3dv.json"


# ---------------GRAPHITS SPARSE_90 (N3DV)---------------

sbatch   --job-name=coffee_martini_graphits_sparse_90   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphits_sparse_90 --config configs/graphits_sparse_views_10_n3dv.json"

sbatch   --job-name=cook_spinach_graphits_sparse_90   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphits_sparse_90 --config configs/graphits_sparse_views_10_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphits_sparse_90   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphits_sparse_90 --config configs/graphits_sparse_views_10_n3dv.json"

sbatch   --job-name=flame_salmon_graphits_sparse_90   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphits_sparse_90 --config configs/graphits_sparse_views_10_n3dv.json"

sbatch   --job-name=flame_steak_graphits_sparse_90   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphits_sparse_90 --config configs/graphits_sparse_views_10_n3dv.json"

sbatch   --job-name=sear_steak_graphits_sparse_90   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphits_sparse_90 --config configs/graphits_sparse_views_10_n3dv.json"


# ---------------GRAPHIGS SPARSE_90 (N3DV)---------------

sbatch   --job-name=coffee_martini_graphigs_sparse_90   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphigs_sparse_90 --config configs/graphigs_sparse_views_10_n3dv.json"

sbatch   --job-name=cook_spinach_graphigs_sparse_90   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphigs_sparse_90 --config configs/graphigs_sparse_views_10_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphigs_sparse_90   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphigs_sparse_90 --config configs/graphigs_sparse_views_10_n3dv.json"

sbatch   --job-name=flame_salmon_graphigs_sparse_90   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphigs_sparse_90 --config configs/graphigs_sparse_views_10_n3dv.json"

sbatch   --job-name=flame_steak_graphigs_sparse_90   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphigs_sparse_90 --config configs/graphigs_sparse_views_10_n3dv.json"

sbatch   --job-name=sear_steak_graphigs_sparse_90   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphigs_sparse_90 --config configs/graphigs_sparse_views_10_n3dv.json"


# ---------------GRAPHITS FPS_10 (N3DV)---------------

sbatch   --job-name=coffee_martini_graphits_fps_10   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphits_fps_10 --config configs/graphits_sparse_frames_10fps_n3dv.json"

sbatch   --job-name=cook_spinach_graphits_fps_10   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphits_fps_10 --config configs/graphits_sparse_frames_10fps_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphits_fps_10   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphits_fps_10 --config configs/graphits_sparse_frames_10fps_n3dv.json"

sbatch   --job-name=flame_salmon_graphits_fps_10   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphits_fps_10 --config configs/graphits_sparse_frames_10fps_n3dv.json"

sbatch   --job-name=flame_steak_graphits_fps_10   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphits_fps_10 --config configs/graphits_sparse_frames_10fps_n3dv.json"

sbatch   --job-name=sear_steak_graphits_fps_10   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphits_fps_10 --config configs/graphits_sparse_frames_10fps_n3dv.json"


# ---------------GRAPHIGS FPS_10 (N3DV)---------------

sbatch   --job-name=coffee_martini_graphigs_fps_10   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphigs_fps_10 --config configs/graphigs_sparse_frames_10fps_n3dv.json"

sbatch   --job-name=cook_spinach_graphigs_fps_10   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphigs_fps_10 --config configs/graphigs_sparse_frames_10fps_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphigs_fps_10   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphigs_fps_10 --config configs/graphigs_sparse_frames_10fps_n3dv.json"

sbatch   --job-name=flame_salmon_graphigs_fps_10   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphigs_fps_10 --config configs/graphigs_sparse_frames_10fps_n3dv.json"

sbatch   --job-name=flame_steak_graphigs_fps_10   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphigs_fps_10 --config configs/graphigs_sparse_frames_10fps_n3dv.json"

sbatch   --job-name=sear_steak_graphigs_fps_10   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphigs_fps_10 --config configs/graphigs_sparse_frames_10fps_n3dv.json"


# ---------------GRAPHITS FPS_20 (N3DV)---------------

sbatch   --job-name=coffee_martini_graphits_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphits_fps_20 --config configs/graphits_sparse_frames_20fps_n3dv.json"

sbatch   --job-name=cook_spinach_graphits_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphits_fps_20 --config configs/graphits_sparse_frames_20fps_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphits_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphits_fps_20 --config configs/graphits_sparse_frames_20fps_n3dv.json"

sbatch   --job-name=flame_salmon_graphits_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphits_fps_20 --config configs/graphits_sparse_frames_20fps_n3dv.json"

sbatch   --job-name=flame_steak_graphits_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphits_fps_20 --config configs/graphits_sparse_frames_20fps_n3dv.json"

sbatch   --job-name=sear_steak_graphits_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphits_fps_20 --config configs/graphits_sparse_frames_20fps_n3dv.json"


# ---------------GRAPHIGS FPS_20 (N3DV)---------------

sbatch   --job-name=coffee_martini_graphigs_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphigs_fps_20 --config configs/graphigs_sparse_frames_20fps_n3dv.json"

sbatch   --job-name=cook_spinach_graphigs_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphigs_fps_20 --config configs/graphigs_sparse_frames_20fps_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphigs_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphigs_fps_20 --config configs/graphigs_sparse_frames_20fps_n3dv.json"

sbatch   --job-name=flame_salmon_graphigs_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphigs_fps_20 --config configs/graphigs_sparse_frames_20fps_n3dv.json"

sbatch   --job-name=flame_steak_graphigs_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphigs_fps_20 --config configs/graphigs_sparse_frames_20fps_n3dv.json"

sbatch   --job-name=sear_steak_graphigs_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphigs_fps_20 --config configs/graphigs_sparse_frames_20fps_n3dv.json"


# ---------------GRAPHITS NOSYNC_50_FPS_20 (N3DV)---------------

sbatch   --job-name=coffee_martini_graphits_nosync_50_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphits_nosync_50_fps_20 --config configs/graphits_unsync_50_n3dv.json"

sbatch   --job-name=cook_spinach_graphits_nosync_50_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphits_nosync_50_fps_20 --config configs/graphits_unsync_50_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphits_nosync_50_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphits_nosync_50_fps_20 --config configs/graphits_unsync_50_n3dv.json"

sbatch   --job-name=flame_salmon_graphits_nosync_50_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphits_nosync_50_fps_20 --config configs/graphits_unsync_50_n3dv.json"

sbatch   --job-name=flame_steak_graphits_nosync_50_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphits_nosync_50_fps_20 --config configs/graphits_unsync_50_n3dv.json"

sbatch   --job-name=sear_steak_graphits_nosync_50_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphits_nosync_50_fps_20 --config configs/graphits_unsync_50_n3dv.json"


# ---------------GRAPHIGS NOSYNC_50_FPS_20 (N3DV)---------------

sbatch   --job-name=coffee_martini_graphigs_nosync_50_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphigs_nosync_50_fps_20 --config configs/graphigs_unsync_50_n3dv.json"

sbatch   --job-name=cook_spinach_graphigs_nosync_50_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphigs_nosync_50_fps_20 --config configs/graphigs_unsync_50_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphigs_nosync_50_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphigs_nosync_50_fps_20 --config configs/graphigs_unsync_50_n3dv.json"

sbatch   --job-name=flame_salmon_graphigs_nosync_50_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphigs_nosync_50_fps_20 --config configs/graphigs_unsync_50_n3dv.json"

sbatch   --job-name=flame_steak_graphigs_nosync_50_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphigs_nosync_50_fps_20 --config configs/graphigs_unsync_50_n3dv.json"

sbatch   --job-name=sear_steak_graphigs_nosync_50_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphigs_nosync_50_fps_20 --config configs/graphigs_unsync_50_n3dv.json"


# ---------------GRAPHITS NOSYNC_90_FPS_20 (N3DV)---------------

sbatch   --job-name=coffee_martini_graphits_nosync_90_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphits_nosync_90_fps_20 --config configs/graphits_unsync_10_n3dv.json"

sbatch   --job-name=cook_spinach_graphits_nosync_90_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphits_nosync_90_fps_20 --config configs/graphits_unsync_10_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphits_nosync_90_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphits_nosync_90_fps_20 --config configs/graphits_unsync_10_n3dv.json"

sbatch   --job-name=flame_salmon_graphits_nosync_90_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphits_nosync_90_fps_20 --config configs/graphits_unsync_10_n3dv.json"

sbatch   --job-name=flame_steak_graphits_nosync_90_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphits_nosync_90_fps_20 --config configs/graphits_unsync_10_n3dv.json"

sbatch   --job-name=sear_steak_graphits_nosync_90_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphits_nosync_90_fps_20 --config configs/graphits_unsync_10_n3dv.json"


# ---------------GRAPHIGS NOSYNC_90_FPS_20 (N3DV)---------------

sbatch   --job-name=coffee_martini_graphigs_nosync_90_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphigs_nosync_90_fps_20 --config configs/graphigs_unsync_10_n3dv.json"

sbatch   --job-name=cook_spinach_graphigs_nosync_90_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphigs_nosync_90_fps_20 --config configs/graphigs_unsync_10_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphigs_nosync_90_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphigs_nosync_90_fps_20 --config configs/graphigs_unsync_10_n3dv.json"

sbatch   --job-name=flame_salmon_graphigs_nosync_90_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphigs_nosync_90_fps_20 --config configs/graphigs_unsync_10_n3dv.json"

sbatch   --job-name=flame_steak_graphigs_nosync_90_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphigs_nosync_90_fps_20 --config configs/graphigs_unsync_10_n3dv.json"

sbatch   --job-name=sear_steak_graphigs_nosync_90_fps_20   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphigs_nosync_90_fps_20 --config configs/graphigs_unsync_10_n3dv.json"


# ---------------GRAPHITS FAULTY_CAMS (N3DV)---------------

sbatch   --job-name=coffee_martini_graphits_faulty_cams   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphits_faulty_cams --config configs/graphits_faulty_cams_1_n3dv.json"

sbatch   --job-name=cook_spinach_graphits_faulty_cams   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphits_faulty_cams --config configs/graphits_faulty_cams_1_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphits_faulty_cams   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphits_faulty_cams --config configs/graphits_faulty_cams_1_n3dv.json"

sbatch   --job-name=flame_salmon_graphits_faulty_cams   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphits_faulty_cams --config configs/graphits_faulty_cams_1_n3dv.json"

sbatch   --job-name=flame_steak_graphits_faulty_cams   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphits_faulty_cams --config configs/graphits_faulty_cams_1_n3dv.json"

sbatch   --job-name=sear_steak_graphits_faulty_cams   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphits_faulty_cams --config configs/graphits_faulty_cams_1_n3dv.json"


# ---------------GRAPHIGS FAULTY_CAMS (N3DV)---------------

sbatch   --job-name=coffee_martini_graphigs_faulty_cams   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphigs_faulty_cams --config configs/graphigs_faulty_cams_1_n3dv.json"

sbatch   --job-name=cook_spinach_graphigs_faulty_cams   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphigs_faulty_cams --config configs/graphigs_faulty_cams_1_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphigs_faulty_cams   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphigs_faulty_cams --config configs/graphigs_faulty_cams_1_n3dv.json"

sbatch   --job-name=flame_salmon_graphigs_faulty_cams   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphigs_faulty_cams --config configs/graphigs_faulty_cams_1_n3dv.json"

sbatch   --job-name=flame_steak_graphigs_faulty_cams   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphigs_faulty_cams --config configs/graphigs_faulty_cams_1_n3dv.json"

sbatch   --job-name=sear_steak_graphigs_faulty_cams   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphigs_faulty_cams --config configs/graphigs_faulty_cams_1_n3dv.json"


# ---------------GRAPHITS FAULTY_CAMS_2 (N3DV)---------------

sbatch   --job-name=coffee_martini_graphits_faulty_cams_2   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphits_faulty_cams_2 --config configs/graphits_faulty_cams_2_n3dv.json"

sbatch   --job-name=cook_spinach_graphits_faulty_cams_2   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphits_faulty_cams_2 --config configs/graphits_faulty_cams_2_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphits_faulty_cams_2   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphits_faulty_cams_2 --config configs/graphits_faulty_cams_2_n3dv.json"

sbatch   --job-name=flame_salmon_graphits_faulty_cams_2   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphits_faulty_cams_2 --config configs/graphits_faulty_cams_2_n3dv.json"

sbatch   --job-name=flame_steak_graphits_faulty_cams_2   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphits_faulty_cams_2 --config configs/graphits_faulty_cams_2_n3dv.json"

sbatch   --job-name=sear_steak_graphits_faulty_cams_2   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphits_faulty_cams_2 --config configs/graphits_faulty_cams_2_n3dv.json"


# ---------------GRAPHIGS FAULTY_CAMS_2 (N3DV)---------------

sbatch   --job-name=coffee_martini_graphigs_faulty_cams_2   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphigs_faulty_cams_2 --config configs/graphigs_faulty_cams_2_n3dv.json"

sbatch   --job-name=cook_spinach_graphigs_faulty_cams_2   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphigs_faulty_cams_2 --config configs/graphigs_faulty_cams_2_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphigs_faulty_cams_2   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphigs_faulty_cams_2 --config configs/graphigs_faulty_cams_2_n3dv.json"

sbatch   --job-name=flame_salmon_graphigs_faulty_cams_2   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphigs_faulty_cams_2 --config configs/graphigs_faulty_cams_2_n3dv.json"

sbatch   --job-name=flame_steak_graphigs_faulty_cams_2   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphigs_faulty_cams_2 --config configs/graphigs_faulty_cams_2_n3dv.json"

sbatch   --job-name=sear_steak_graphigs_faulty_cams_2   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphigs_faulty_cams_2 --config configs/graphigs_faulty_cams_2_n3dv.json"


# ---------------GRAPHITS FAULTY_CAMS_3 (N3DV)---------------

sbatch   --job-name=coffee_martini_graphits_faulty_cams_3   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphits_faulty_cams_3 --config configs/graphits_faulty_cams_3_n3dv.json"

sbatch   --job-name=cook_spinach_graphits_faulty_cams_3   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphits_faulty_cams_3 --config configs/graphits_faulty_cams_3_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphits_faulty_cams_3   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphits_faulty_cams_3 --config configs/graphits_faulty_cams_3_n3dv.json"

sbatch   --job-name=flame_salmon_graphits_faulty_cams_3   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphits_faulty_cams_3 --config configs/graphits_faulty_cams_3_n3dv.json"

sbatch   --job-name=flame_steak_graphits_faulty_cams_3   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphits_faulty_cams_3 --config configs/graphits_faulty_cams_3_n3dv.json"

sbatch   --job-name=sear_steak_graphits_faulty_cams_3   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphits_faulty_cams_3 --config configs/graphits_faulty_cams_3_n3dv.json"


# ---------------GRAPHIGS FAULTY_CAMS_3 (N3DV)---------------

sbatch   --job-name=coffee_martini_graphigs_faulty_cams_3   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/coffee_martini -m output/coffee_martini_graphigs_faulty_cams_3 --config configs/graphigs_faulty_cams_3_n3dv.json"

sbatch   --job-name=cook_spinach_graphigs_faulty_cams_3   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cook_spinach -m output/cook_spinach_graphigs_faulty_cams_3 --config configs/graphigs_faulty_cams_3_n3dv.json"

sbatch   --job-name=cut_roasted_beef_graphigs_faulty_cams_3   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/cut_roasted_beef -m output/cut_roasted_beef_graphigs_faulty_cams_3 --config configs/graphigs_faulty_cams_3_n3dv.json"

sbatch   --job-name=flame_salmon_graphigs_faulty_cams_3   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_salmon -m output/flame_salmon_graphigs_faulty_cams_3 --config configs/graphigs_faulty_cams_3_n3dv.json"

sbatch   --job-name=flame_steak_graphigs_faulty_cams_3   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/flame_steak -m output/flame_steak_graphigs_faulty_cams_3 --config configs/graphigs_faulty_cams_3_n3dv.json"

sbatch   --job-name=sear_steak_graphigs_faulty_cams_3   --gres=gpu:1   --time=24:00:00   --output=logs/%x_%j.out   --error=logs/%x_%j.err   --wrap="python train.py -s data/N3DV/sear_steak -m output/sear_steak_graphigs_faulty_cams_3 --config configs/graphigs_faulty_cams_3_n3dv.json"
