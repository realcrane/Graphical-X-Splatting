#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This file is a derivative work of the original software.
# Modifications and additions:
# 2026, Doga Yilmaz (doga.yilmaz@ucl.ac.uk)
# Virtual Environments and Computer Graphics Lab, UCL
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For original inquiries contact  george.drettakis@inria.fr
# For modification inquiries contact doga.yilmaz@ucl.ac.uk
#

import torch
from scene import Scene_nt
import os
import subprocess
import torchvision
from tqdm import tqdm
from os import makedirs
import numpy as np
from graphixs_renderer import render
from utils.system_utils import safe_state
from utils.image_utils import easy_cmap, make_grid_w_props, percentile_magnitude_cmap
from scene.cameras import CameraImmersive
from utils.fisheye_utils import apply_distortion_warp
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams
from scene.nt_model import NTModel
import sys
import json


def benchmark_render(model_path, scene, pipeline, background, num_renders=50, warmup_renders=25):
    """Benchmark rendering FPS and GPU memory usage without saving images."""
    print("\n=== Benchmark Mode ===")
    print(f"Rendering {num_renders} frames ({warmup_renders} warmup)...")
    
    # Get all available cameras
    cams = scene.getTestCameras()
    if not cams:
        cams = scene.getTrainCameras()
    
    if not cams:
        print("No cameras found for benchmarking!")
        return
    
    # Collect frames from all cameras
    all_frames = []
    for cam_id, cam in cams.items():
        for frame_id, frame in cam.items():
            all_frames.append(frame)
    
    if len(all_frames) == 0:
        print("No frames found for benchmarking!")
        return
    
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    
    render_times = []
    
    with torch.no_grad():
        for i in range(num_renders):
            frame = all_frames[i % len(all_frames)]
            
            start_evt = torch.cuda.Event(enable_timing=True)
            end_evt = torch.cuda.Event(enable_timing=True)
            
            start_evt.record()
            render_pack = render(frame, scene.primitives, pipeline, background, inference_only=True)
            end_evt.record()
            
            torch.cuda.synchronize()
            elapsed_time = start_evt.elapsed_time(end_evt)  # milliseconds
            
            # Only record times after warmup
            if i >= warmup_renders:
                render_times.append(elapsed_time)
            
            # Progress indicator
            if (i + 1) % 10 == 0:
                print(f"Rendered {i + 1}/{num_renders} images...")
    
    # GPU memory stats
    peak_mem_bytes = torch.cuda.max_memory_allocated()
    peak_mem_mb = peak_mem_bytes / (1024 ** 2)
    
    # FPS stats
    avg_time_ms = np.mean(render_times)
    std_time_ms = np.std(render_times)
    fps = 1000.0 / avg_time_ms
    
    results = {
        "num_renders": num_renders,
        "warmup_renders": warmup_renders,
        "measured_renders": len(render_times),
        "avg_render_time_ms": float(avg_time_ms),
        "std_render_time_ms": float(std_time_ms),
        "fps": float(fps),
        "peak_gpu_memory_mb": float(peak_mem_mb),
        "peak_gpu_memory_bytes": int(peak_mem_bytes),
    }

    out_path = os.path.join(model_path, "benchmark_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nAvg render time: {avg_time_ms:.2f} +/- {std_time_ms:.2f} ms")
    print(f"FPS: {fps:.2f}")
    print(f"Peak GPU memory: {peak_mem_mb:.2f} MB")
    print(f"Results saved to {out_path}")
    print("======================\n")

def render_set(model_path, name, iteration, scene, pipeline, background, render_videos):
    if name == "train":
        cams = scene.getTrainCameras()
    elif name == "test":
        cams = scene.getTestCameras()
    else:
        raise ValueError(f"Unknown dataset name: {name}")

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    grid_path = os.path.join(model_path, name, "ours_{}".format(iteration), "grids")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(grid_path, exist_ok=True)

    progress_bar = tqdm(total=sum(len(frames) for frames in cams.values()), desc="Rendering", ncols=80)
    render_avg_time = 0.0
    idx = 0
    for cam_id, cam in cams.items():
        cam_render_path = os.path.join(render_path, f"cam_{cam_id}")
        cam_gts_path = os.path.join(gts_path, f"cam_{cam_id}")
        cam_grid_path = os.path.join(grid_path, f"cam_{cam_id}")
        makedirs(cam_render_path, exist_ok=True)
        makedirs(cam_gts_path, exist_ok=True)
        makedirs(cam_grid_path, exist_ok=True)

        for frame_id, frame in cam.items():
            progress_bar.set_description(f"Rendering C:{cam_id} F:{frame_id}")
            render_start_event = torch.cuda.Event(enable_timing=True)
            render_end_event = torch.cuda.Event(enable_timing=True)
            render_start_event.record()
            render_pack = render(frame, scene.primitives, pipeline, background)
            render_end_event.record()
            
            rendering = torch.clamp(render_pack["render"], 0.0, 1.0)
            gt_image = torch.clamp(frame.original_image.to("cuda"), 0.0, 1.0)

            # Immersive: warp 2× rendered image through fisheye distortion flow
            if isinstance(frame, CameraImmersive) and frame.fisheyemapper is not None:
                rendering = apply_distortion_warp(rendering, frame.fisheyemapper,
                                                  frame.gt_height, frame.gt_width)
                rendering = torch.clamp(rendering, 0.0, 1.0)

            rendered_likelihoods = percentile_magnitude_cmap(render_pack["rendered_likelihoods"][0], colormap='inferno')
            rendered_velocity = percentile_magnitude_cmap(render_pack["rendered_velocity"][0], colormap='inferno')
            rendered_acceleration = percentile_magnitude_cmap(render_pack["rendered_acceleration"][0], colormap='inferno')
            rendered_jerk = percentile_magnitude_cmap(render_pack["rendered_jerk"][0], colormap='inferno')
            rendered_snap = percentile_magnitude_cmap(render_pack["rendered_snap"][0], colormap='inferno')
            
            grid = [gt_image, rendering, rendered_likelihoods, rendered_velocity, rendered_acceleration, rendered_jerk, rendered_snap]
            grid = make_grid_w_props(grid, ["GT", "Render", "Likelihood", "Velocity", "Acceleration", "Jerk", "Snap"])

            torch.cuda.synchronize()
            render_avg_time += render_start_event.elapsed_time(render_end_event)

            out_render_path = os.path.join(cam_render_path, f"{frame_id}.png")
            out_gt_path = os.path.join(cam_gts_path, f"{frame_id}.png")
            out_grid_path = os.path.join(cam_grid_path, f"{frame_id}_grid.png")
            torchvision.utils.save_image(rendering, out_render_path)
            torchvision.utils.save_image(gt_image, out_gt_path)
            torchvision.utils.save_image(grid, out_grid_path)
            frame.unload_image()
            progress_bar.update(1)
            idx += 1

        if render_videos:            
            render_input_pattern = os.path.join(cam_render_path, "%d.png")
            render_output_video = os.path.join(cam_render_path, f"cam_{cam_id}.mp4")
            cmd = [
                "ffmpeg",
                "-nostats", "-loglevel", "0",
                "-y",
                "-framerate", "30",
                "-i", render_input_pattern,
                "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
                "-c", "libx264",
                "-pix_fmt", "yuv420p",
                render_output_video,
            ]
            try:
                subprocess.run(cmd, check=True)
            except Exception as e:
                print(f"ffmpeg video creation failed for camera {cam_id}: {e}")

            gt_input_pattern = os.path.join(cam_gts_path, "%d.png")
            gt_output_video = os.path.join(cam_gts_path, f"cam_{cam_id}_gt.mp4")
            cmd = [
                "ffmpeg",
                "-nostats", "-loglevel", "0",
                "-y",
                "-framerate", "30",
                "-i", gt_input_pattern,
                "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
                "-c", "libx264",
                "-pix_fmt", "yuv420p",
                gt_output_video,
            ]
            try:
                subprocess.run(cmd, check=True)
            except Exception as e:
                print(f"ffmpeg video creation failed for camera {cam_id} ground truth: {e}")
            
            grid_input_pattern = os.path.join(cam_grid_path, "%d_grid.png")
            grid_output_video = os.path.join(cam_grid_path, f"cam_{cam_id}_grid.mp4")
            cmd = [
                "ffmpeg",
                "-nostats", "-loglevel", "0",
                "-y",
                "-framerate", "30",
                "-i", grid_input_pattern,
                "-vf", "pad=width=ceil(iw/2)*2:height=ceil(ih/2)*2",
                "-c", "libx264",
                "-pix_fmt", "yuv420p",
                grid_output_video,
            ]
            try:
                subprocess.run(cmd, check=True)
            except Exception as e:
                print(f"ffmpeg video creation failed for camera {cam_id} grid: {e}")

    progress_bar.close()

    print("average render time: ", render_avg_time/1000/idx)
    print("FPS:", idx/(render_avg_time/1000.0))

def find_checkpoint(model_path , preferred_iter=None):
    """Return path to checkpoint matching preferred_iter if present, else latest available, else None."""
    if preferred_iter is not None:
        cand = os.path.join(model_path, f"chkpnt{preferred_iter}.pth")
        if os.path.isfile(cand):
            print(f"Found preferred checkpoint {cand}")
            return cand
        else:
            print(f"Preferred checkpoint {cand} not found, looking for latest available")

    # Find latest checkpoint
    latest_iter = -1
    for fname in os.listdir(model_path):
        if fname.startswith("chkpnt") and fname.endswith(".pth"):
            try:
                it = int(fname[len("chkpnt"):-len(".pth")])
                if it > latest_iter:
                    latest_iter = it
            except ValueError:
                continue

    if latest_iter >= 0:
        latest_cand = os.path.join(model_path, f"chkpnt{latest_iter}.pth")
        print(f"Using latest checkpoint {latest_cand}")
        return latest_cand

    print("No checkpoints found in", model_path)
    return None

def render_sets(dataset : ModelParams, iteration : str, pipeline : PipelineParams, skip_train : bool, skip_test : bool, render_videos : bool = False, benchmark : bool = False):
    with torch.no_grad():
        # Mirror NTModel initialization from training to respect config flags
        primitives = NTModel(
            dataset.sh_degree,
            dataset.nu_degree,
            args=args,

        )
        scene = Scene_nt(dataset, primitives, load_iteration=0, shuffle=False)

        checkpoint = find_checkpoint(dataset.model_path, iteration)
        if checkpoint is None:
            print(f"ERROR: No checkpoint found in {dataset.model_path}. Cannot render.")
            return
        first_iter = scene.load(checkpoint, is_train=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if benchmark:
            benchmark_render(dataset.model_path, scene, pipeline, background)
            return

        if not skip_train:
            render_set(dataset.model_path, "train", first_iter, scene, pipeline, background, render_videos)

        if not skip_test:
            render_set(dataset.model_path, "test", first_iter, scene, pipeline, background, render_videos)

        nu_degree = scene.primitives.get_nu_degree

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    parser = ArgumentParser(description="Rendering script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default="40000", type=str)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--render-videos", action="store_true")
    parser.add_argument("--benchmark", action="store_true", help="Benchmark FPS and GPU memory instead of rendering images")
    parser.add_argument('--config', type=str, default=None)

    args = parser.parse_args(sys.argv[1:])

    merged: dict = {}
    if getattr(args, 'model_path', None):
        cfgfilepath = os.path.join(args.model_path, "cfg_args")
        try:
            with open(cfgfilepath) as cfg_file:
                cfgfile_string = cfg_file.read()
                args_cfgfile = eval(cfgfile_string)
                merged.update(vars(args_cfgfile))
        except Exception:
            print("Config file not found at", cfgfilepath)

    if args.config is not None:
        config = load_config(args.config)
        merged.update(config)

    for k, v in vars(args).items():
        if v is not None:
            merged[k] = v

    args = Namespace(**merged)

    safe_state(args.quiet)

    print("Rendering " + args.model_path)
    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test, args.render_videos, args.benchmark)