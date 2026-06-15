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

import os
import json
import subprocess
import shutil
import torch
import numpy as np
import random
from random import randint
from utils.loss_utils import l1_loss, ssim, temporal_opacity_regularization
from graphixs_renderer import render
import sys
from scene import Scene_nt, NTModel
from utils.general_utils import op_sigmoid
from utils.system_utils import safe_state
import uuid
from tqdm import tqdm
from utils.image_utils import psnr, easy_cmap, make_grid_w_props, percentile_magnitude_cmap
from scene.cameras import CameraImmersive
from utils.fisheye_utils import apply_distortion_warp
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from scene.nt_model import build_scaling_rotation
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

from os import makedirs
import torchvision
from PIL import Image
import torchvision.transforms.functional as tf

from utils.general_utils import get_expon_lr_func

def build_diff_graphixs_rasterization():
    extension_dir = os.path.join('submodules', 'diff-graphixs-rasterization')
    build_dir = os.path.join(extension_dir, 'build')
    if os.path.exists(build_dir):
        shutil.rmtree(build_dir)
    subprocess.check_call(
        [sys.executable, 'setup.py', 'build_ext', '--inplace'],
        cwd=extension_dir
    )

def training(dataset, opt, pipe, testing_iterations, checkpoint_iterations, checkpoint, debug_from, disable_tensorboard=False):
    if dataset.cap_max == -1:
        print("Please specify the maximum number of Gaussians using --cap_max.")
        exit()
    first_iter = 0
    merged_args = Namespace(**vars(dataset), **vars(opt), **vars(pipe))
    tb_writer = prepare_output_and_logger(merged_args, disable_tensorboard)
    primitives = NTModel(dataset.sh_degree, 
                         dataset.nu_degree,
                         args=args)
    scene = Scene_nt(dataset, primitives)
    """
    NOTE: 
    This part of code is tricky,
    the setting for learning rate is different with first and second order optimizer.
    For second order optimizer, the real learning rate is approximately lr^2.
    We are adjusting suitable learning rate for our second order optimizer here.
    This is made based on real learning rate (base_lr * spatial_lr_scale) of Train scene from Tanks&Temples dataset.
    The '0.001226442339330813' is the base_lr we want for start,
    and the '1.226630619638022e-05' is the base_lr we at the end.
    """
    train_scene_scale = 7.45176315307617
    opt.position_lr_init = pow(0.001226442339330813*primitives.spatial_lr_scale/train_scene_scale, 0.5)/primitives.spatial_lr_scale
    opt.position_lr_final = pow(1.226630619638022e-05*primitives.spatial_lr_scale/train_scene_scale, 0.5)/primitives.spatial_lr_scale

    xyz_lr_sqrt_args = get_expon_lr_func(lr_init=opt.position_lr_init*primitives.spatial_lr_scale,
                                                    lr_final=opt.position_lr_final*primitives.spatial_lr_scale,
                                                    lr_delay_mult=opt.position_lr_delay_mult,
                                                    max_steps=opt.position_lr_max_steps)

    print("spatial lr scale: {}".format(primitives.spatial_lr_scale))
    print("training lr range: {} - {}".format(pow(xyz_lr_sqrt_args(1),2), pow(xyz_lr_sqrt_args(opt.position_lr_max_steps),2)))

    C_burnin = opt.C_burnin
    C = opt.C
    burnin_iterations = opt.burnin_iterations
    optimizer_type = opt.optimizer_type
    optimizer_noise_scale = opt.optimizer_noise_scale

    primitives.training_setup(opt, C_burnin, C, burnin_iterations, optimizer_noise_scale, optimizer_type)
    if checkpoint:
        first_iter = scene.load(checkpoint)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    torch.cuda.reset_peak_memory_stats()

    viewpoint_stack = None
    ema_loss_for_log = 0.0
    first_iter += 1

    BATCH_SIZE = opt.batch_size

    viewpoint_stack = scene.getTrainCameras().copy()

    best_test_loss = 0.0
    best_train_loss = 0.0
    print(f"[SAVE_ITERS] {' '.join(str(i) for i in sorted(set(checkpoint_iterations)))}", flush=True)
    for iteration in range(first_iter, opt.iterations + 1):
        iter_start.record()
        primitives.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            primitives.oneupSHdegree()

        batch_loss = 0.0
        batch_component_conf = None
        batch_visibility_filter = torch.zeros(primitives.get_xyz.shape[0], dtype=torch.bool, device="cuda")
        for i in range(BATCH_SIZE):
            # Pick a random Camera
            if not viewpoint_stack:
                viewpoint_stack = scene.getTrainCameras().copy()

            random_cam_key = list(viewpoint_stack.keys())[randint(0, len(viewpoint_stack)-1)]
            viewpoint_cam = viewpoint_stack.pop(random_cam_key)
            random_frame_key = list(viewpoint_cam.keys())[randint(0, len(viewpoint_cam)-1)]
            viewpoint_frame = viewpoint_cam[random_frame_key]

            # Render
            if (iteration - 1) == debug_from:
                pipe.debug = True

            bg = torch.rand((3), device="cuda") if opt.random_background else background

            render_pkg = render(viewpoint_frame, primitives, pipe, bg, dynamics_noise_std=opt.dynamics_noise_std)
            image = render_pkg["render"]
            likelihoods = render_pkg["likelihoods"]
            primitives.likelihoods += likelihoods.detach().clone()

            # Immersive: warp 2× rendered image through fisheye distortion flow
            if isinstance(viewpoint_frame, CameraImmersive) and viewpoint_frame.fisheyemapper is not None:
                image = apply_distortion_warp(image, viewpoint_frame.fisheyemapper,
                                              viewpoint_frame.gt_height, viewpoint_frame.gt_width)
            
            # Accumulate visibility across all views in batch
            batch_visibility_filter = torch.logical_or(batch_visibility_filter, render_pkg["radii"] > 0.0)

            # Accumulate likelihoods across cameras for P(Z | G) calculation
            if batch_component_conf is None:
                batch_component_conf = likelihoods.clone()
            else:
                batch_component_conf += likelihoods

            # Loss - Negative Log-Likelihood formulation
            # Rendering likelihood: -log P(I_T | G, C, T)
            gt_image = viewpoint_frame.original_image.cuda()
            L_render_l1 = (1.0 - opt.lambda_dssim) * l1_loss(image, gt_image)
            L_render_ssim = opt.lambda_dssim * (1.0 - ssim(image, gt_image))
            # L_img regularizers (eq. imageDis): opacity-L1 and sqrt-eigenvalue-L1.
            # For this parameterization sqrt(eigenvalue of Sigma) == scaling.
            L_opacity_l1 = opt.opacity_l1_reg * primitives.get_base_opacity.mean()
            L_eigen_l1 = opt.eigenvalue_l1_reg * primitives.get_scaling.mean()
            L_X = L_render_l1 + L_render_ssim + L_opacity_l1 + L_eigen_l1
            
            # Prior likelihood: -log P(G_T | C, T)
            # Temporal opacity regularization as part of dynamics prior
            temporal_reg_loss = temporal_opacity_regularization(
                    primitives.get_base_opacity,
                    primitives.get_origin_time,
                    primitives.get_duration,
                    viewpoint_frame.timestamp,
                    likelihoods.detach()
            )
            L_opacity = args.temporal_opacity_reg * temporal_reg_loss
            
            # Active set: components contributing non-negligibly to the render at time t,
            # weighted softly by their per-component response (occlusion-aware soft visibility).
            w = likelihoods.detach()
            w_sum = w.sum() + opt.likelihood_epsilon

            # Shape prior P(Sigma): pull each component's covariance toward the
            # contribution-weighted mean covariance Sigma_hat^t (eq. shapePrior).
            scales = primitives.get_scaling
            L_cov = build_scaling_rotation(scales, primitives.get_rotation)
            cov = L_cov @ L_cov.transpose(1, 2)                          # (N, 3, 3)
            cov_mean = (w[:, None, None] * cov).sum(dim=0) / w_sum       # Sigma_hat^t
            fro_sq = ((cov - cov_mean[None]) ** 2).sum(dim=(-1, -2))     # ||.||_F^2 per comp
            scale_reg_loss = (w * fro_sq).sum() / w_sum
            L_scale = opt.scale_reg * scale_reg_loss

            # Motion prior P(v,a,j,s): volume-weighted, normalized over the same active set
            # (eq. motionPrior). Volume proxy prod|scale| == sqrt|det Sigma|.
            gaussian_volumes = torch.prod(torch.abs(scales), dim=-1)
            motion_weight = w * gaussian_volumes

            velocity_magnitude = torch.norm(primitives.get_velocity, dim=-1)
            L_velocity = args.velocity_reg * (motion_weight * velocity_magnitude ** 2).sum() / w_sum

            acceleration_magnitude = torch.norm(primitives.get_acceleration, dim=-1)
            L_acceleration = args.acceleration_reg * (motion_weight * acceleration_magnitude ** 2).sum() / w_sum

            jerk_magnitude = torch.norm(primitives.get_jerk, dim=-1)
            L_jerk = args.jerk_reg * (motion_weight * jerk_magnitude ** 2).sum() / w_sum

            snap_magnitude = torch.norm(primitives.get_snap, dim=-1)
            L_snap = args.snap_reg * (motion_weight * snap_magnitude ** 2).sum() / w_sum
            
            batch_loss += L_X + L_opacity + L_scale + L_velocity + L_acceleration + L_jerk + L_snap
            
            viewpoint_frame.unload_image()

        # Average negative log-likelihood over batch
        batch_loss = batch_loss / BATCH_SIZE
        
        # This encourages primitives to be visible across multiple cameras
        numerators = batch_component_conf + opt.likelihood_epsilon
        denominator = torch.sum(batch_component_conf) + opt.likelihood_epsilon
        L_component_conf = opt.likelihood_reg * torch.sum(torch.log(denominator) - torch.log(numerators)) / primitives.get_xyz.shape[0]
        
        total_loss = batch_loss + L_component_conf
                
        total_loss.backward()
        
        iter_end.record()
        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * total_loss.item() + 0.6 * ema_loss_for_log
            if iteration % 10 == 0 or iteration == opt.iterations:
                n_gauss = primitives.get_xyz.shape[0]
                print(f"[ITER {iteration}/{opt.iterations}] "
                      f"loss={ema_loss_for_log:.6f} gaussians={n_gauss}", flush=True)
            
            # Use accumulated visibility from all views in batch
            primitives.add_densification_stats(batch_visibility_filter)

            # Optimizer step
            if iteration < opt.iterations:
                if primitives.optimizer_type.lower() == "sghmc":
                    # NOTE: SGHMC optimization
                    sig = (op_sigmoid(1 - primitives.get_base_opacity))

                    L = build_scaling_rotation(primitives.get_scaling, primitives.get_rotation)
                    actual_covariance = L @ L.transpose(1, 2)
                    
                    primitives.optimizer.step(sig=sig.detach(), cov=actual_covariance.detach())
                else:
                    # Standard Adam optimization
                    primitives.optimizer.step()
                
                primitives.optimizer.zero_grad(set_to_none=True)

            # Log and save
            curr_test_psnr, curr_train_psnr = training_report(tb_writer, iteration, L_render_l1, batch_loss, L_component_conf, L_scale, L_opacity, L_velocity, L_acceleration, L_jerk, L_snap, total_loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background))

            if iteration < opt.densify_until_iter and iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                dead_mask = (primitives.get_base_opacity < opt.opacity_threshold).squeeze(-1)

                #if iteration > opt.prune_low_z_iter:
                #    likelihoods = primitives.likelihoods / (primitives.likelihoods.numel() / primitives.get_xyz.shape[0])
                #    low_likelihood_mask = (likelihoods < (opt.prune_low_z_threshold))
                #    dead_mask = torch.logical_or(dead_mask, low_likelihood_mask)

                primitives.recycle_components_temporal(dead_mask=dead_mask, lambda_g=args.recycle_lambda_g, lambda_o=args.recycle_lambda_o)
                primitives.add_components_temporal(cap_max=args.cap_max, lambda_g=args.recycle_lambda_g, lambda_o=args.recycle_lambda_o)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                scene.save(iteration)

            if iteration in testing_iterations:
                if curr_test_psnr >= best_test_loss:
                    best_test_loss = curr_test_psnr
                    print(f"[ITER {iteration}] New best test PSNR: {curr_test_psnr:.4f}")
                    scene.save(iteration, is_best="test")

                if curr_train_psnr >= best_train_loss:
                    best_train_loss = curr_train_psnr
                    print(f"[ITER {iteration}] New best train PSNR: {curr_train_psnr:.4f}")
                    scene.save(iteration, is_best="train")

    peak_mem = torch.cuda.max_memory_allocated() / (1024 ** 2)
    print(f"\nMax GPU memory usage: {peak_mem:.2f} MB")

def prepare_output_and_logger(args, disable_tensorboard=False):
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    
    # Save all configuration parameters as JSON
    config_dict = vars(args)
    with open(os.path.join(args.model_path, "cfg_args.json"), 'w') as cfg_log_f:
        json.dump(config_dict, cfg_log_f, indent=4, default=str)
    
    # Also save as plain text for backward compatibility
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**config_dict)))

    # Create Tensorboard writer
    tb_writer = None
    if disable_tensorboard:
        print("TensorBoard logging disabled via --disable_tensorboard.")
    elif TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("[WARNING] TensorBoard is not available (install it to enable logging); proceeding without TensorBoard logging.")
    return tb_writer

def training_report(tb_writer, iteration, L_render_l1, batch_loss, L_conf, L_scale, L_opacity, L_velocity, L_acceleration, L_jerk, L_snap, total_loss, l1_loss, elapsed, testing_iterations, scene : Scene_nt, renderFunc, renderArgs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', L_render_l1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/batch_loss', batch_loss.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/conf_regularizer', L_conf.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/scale_regularizer', L_scale.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/opacity_regularizer', L_opacity.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/velocity_regularizer', L_velocity.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/acceleration_regularizer', L_acceleration.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/jerk_regularizer', L_jerk.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/snap_regularizer', L_snap.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', total_loss.item(), iteration)
        tb_writer.add_scalar('stats/iter_time', elapsed, iteration)
        tb_writer.add_scalar('stats/gpu_memory_usage', torch.cuda.memory_allocated(torch.cuda.current_device()), iteration)

    curr_test_psnr = 0.0
    curr_train_psnr = 0.0
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        train_cameras = scene.getTrainCameras()
        train_camera_keys = list(train_cameras.keys())
        selected_train_cameras = {key: train_cameras[key] for idx in range(5, 30, 5) 
                                 if (key := train_camera_keys[idx % len(train_camera_keys)]) in train_cameras}
        
        test_cameras = scene.getTestCameras()
        test_camera_keys = list(test_cameras.keys())
        selected_test_cameras = {key: test_cameras[key] for idx in range(5, 30, 5) 
                                 if (key := test_camera_keys[idx % len(test_camera_keys)]) in test_cameras}
        
        validation_configs = ({'name': 'test', 'cameras' : selected_test_cameras}, 
                              {'name': 'train', 'cameras' : selected_train_cameras})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                test_count = 0
                for idx, viewpoint in enumerate(config['cameras'].values()):
                    for frame_no in args.test_frames:
                        if frame_no not in viewpoint.keys():
                            print(f"[WARNING] {config['name']} camera {idx} does not have frame {frame_no}, skipping.")
                        else:
                            test_count += 1
                            viewpoint_frame = viewpoint[frame_no]
                            render_pack = renderFunc(viewpoint_frame, scene.primitives, *renderArgs)
                            image = torch.clamp(render_pack["render"], 0.0, 1.0)
                            gt_image = torch.clamp(viewpoint_frame.original_image.to("cuda"), 0.0, 1.0)

                            # Immersive: warp 2× rendered image through fisheye distortion flow
                            if isinstance(viewpoint_frame, CameraImmersive) and viewpoint_frame.fisheyemapper is not None:
                                image = apply_distortion_warp(image, viewpoint_frame.fisheyemapper,
                                                              viewpoint_frame.gt_height, viewpoint_frame.gt_width)
                                image = torch.clamp(image, 0.0, 1.0)
                            rendered_likelihoods = percentile_magnitude_cmap(render_pack["rendered_likelihoods"][0], colormap='inferno')
                            rendered_velocity = percentile_magnitude_cmap(render_pack["rendered_velocity"][0], colormap='inferno')
                            rendered_acceleration = percentile_magnitude_cmap(render_pack["rendered_acceleration"][0], colormap='inferno')
                            rendered_jerk = percentile_magnitude_cmap(render_pack["rendered_jerk"][0], colormap='inferno')
                            rendered_snap = percentile_magnitude_cmap(render_pack["rendered_snap"][0], colormap='inferno')
                            if tb_writer and (idx < 5):
                                grid = [gt_image, image, rendered_likelihoods, rendered_velocity, rendered_acceleration, rendered_jerk, rendered_snap]
                                grid = make_grid_w_props(grid, ["GT", "Render", "Likelihood", "Velocity", "Acceleration", "Jerk", "Snap"])
                                tb_writer.add_images(config['name'] + "_view_{}_frame_{}/grid".format(viewpoint_frame.image_name, frame_no), grid[None], global_step=iteration)

                            l1_test += l1_loss(image, gt_image).mean().double()

                            """
                            NOTE: save internal results first, then load saved images to calculate PSNR.
                            This may seem silly, but it is the only way to get the same PSNR as using 'metric.py'.
                            The reason is that the 'metric.py' scipt loads saved image, so the calculation is done with integer type.
                            Without saving, it is calculated with float type.
                            My experience is that you can get a higher PSNR without saving first (some work actually used this trick...).
                            """
                            render_path = os.path.join(scene.model_path, "imgs", "ours_latest", config['name'], "renders")
                            gts_path = os.path.join(scene.model_path, "imgs", "ours_latest", config['name'], "gt")
                            makedirs(render_path, exist_ok=True)
                            makedirs(gts_path, exist_ok=True)
                            save_name = 'cam{:04d}_frame{:04d}'.format(idx, frame_no)
                            torchvision.utils.save_image(image, os.path.join(render_path, save_name + ".png"))
                            torchvision.utils.save_image(gt_image, os.path.join(gts_path, save_name + ".png"))

                            render = Image.open(os.path.join(render_path, save_name + ".png"))
                            gt = Image.open(os.path.join(gts_path, save_name + ".png"))
                            render = tf.to_tensor(render).unsqueeze(0)[:, :3, :, :].cuda()
                            gt = tf.to_tensor(gt).unsqueeze(0)[:, :3, :, :].cuda()
                            psnr_test += psnr(render, gt).mean().double()

                psnr_test /= test_count
                l1_test /= test_count         
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))

                if config['name'] == 'test':
                    curr_test_psnr = psnr_test.item()
                elif config['name'] == 'train':
                    curr_train_psnr = psnr_test.item()
                else:
                    print("[WARNING] Unknown dataset name during evaluation.")

                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
        
        nu_degree = scene.primitives.get_nu_degree
        print("degree of freedom: max: {} min: {} mean: {} std: {}".format(nu_degree.max(), nu_degree.min(), nu_degree.mean(), nu_degree.std()))
        
        if tb_writer:
            near_zero_opacity = torch.sum(scene.primitives.get_base_opacity <= 0.05)
            tb_writer.add_histogram("scene/opacity_histogram", scene.primitives.get_base_opacity, iteration)
            tb_writer.add_histogram("scene/nu_degree_histogram", scene.primitives.get_nu_degree, iteration)
            tb_writer.add_scalar('scene/total_points', scene.primitives.get_xyz.shape[0], iteration)
            tb_writer.add_scalar('scene/near_zero_opacity', near_zero_opacity, iteration)

    return curr_test_psnr, curr_train_psnr

def load_config(config_file):
    with open(config_file, 'r') as file:
        config = json.load(file)
    return config

if __name__ == "__main__":
    # build_diff_graphixs_rasterization()
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--config', type=str, default=None)
    parser.add_argument('--seed', type=int, default=0, help='Random seed for reproducibility')
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument('--disable_tensorboard', action='store_true', default=False, help="Disable TensorBoard logging even if it is installed.")
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[10_000, 20_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[10_000, 20_000, 30_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    
    if args.config is not None:
        cli_tokens = sys.argv[1:]
        cli_provided = set()
        for action in parser._actions:
            for opt in action.option_strings:
                if opt in cli_tokens or any(tok.startswith(opt + "=") for tok in cli_tokens):
                    cli_provided.add(action.dest)
                    break
        
        config = load_config(args.config)

        for key, value in config.items():
            if key in cli_provided:
                continue
            setattr(args, key, value)
    
    
    args.test_iterations = args.test_iterations + [i for i in range(0, op.iterations, 500)]
    args.test_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet, args.seed)

    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args.disable_tensorboard)

    # All done
    print("\nTraining complete.")
