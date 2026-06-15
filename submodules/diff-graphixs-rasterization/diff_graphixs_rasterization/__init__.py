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

from typing import NamedTuple
import torch.nn as nn
import torch
from . import _C

def cpu_deep_copy_tuple(input_tuple):
    copied_tensors = [item.cpu().clone() if isinstance(item, torch.Tensor) else item for item in input_tuple]
    return tuple(copied_tensors)

def rasterize_graphixs(
    means3D,
    means2D,
    sh,
    colors_precomp,
    prev_likelihoods,
    opacities,
    scales,
    rotations,
    cov3Ds_precomp,
    raster_settings,
    nu_degree,
    time,
    duration,
    velocity,
    acceleration,
    jerk,
    snap,
    noise_std
):
    return _RasterizeGraphixs.apply(
        means3D,
        means2D,
        sh,
        colors_precomp,
        prev_likelihoods,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        nu_degree,
        time,
        duration,
        velocity,
        acceleration,
        jerk,
        snap,
        noise_std,
        raster_settings.inference_only
    )

class _RasterizeGraphixs(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        means3D,
        means2D,
        sh,
        colors_precomp,
        prev_likelihoods,
        opacities,
        scales,
        rotations,
        cov3Ds_precomp,
        raster_settings,
        nu_degree,
        time,
        duration,
        velocity,
        acceleration,
        jerk,
        snap,
        noise_std,
        inference_only
    ):

        # Restructure arguments the way that the C++ lib expects them
        args = (
            raster_settings.bg, 
            means3D,
            colors_precomp,
            prev_likelihoods,
            opacities,
            scales,
            rotations,
            raster_settings.scale_modifier,
            cov3Ds_precomp,
            raster_settings.viewmatrix,
            raster_settings.projmatrix,
            raster_settings.tanfovx,
            raster_settings.tanfovy,
            raster_settings.image_height,
            raster_settings.image_width,
            sh,
            raster_settings.sh_degree,
            raster_settings.campos,
            raster_settings.prefiltered,
            raster_settings.debug,
            nu_degree,
            time,
            duration,
            velocity,
            acceleration,
            jerk,
            snap,
            raster_settings.camera_timestamp,
            noise_std,
            inference_only
        )

        # Invoke C++/CUDA rasterizer
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                num_rendered, color, rendered_likelihoods, render_velocity, render_acceleration, render_jerk, render_snap, likelihoods, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_graphixs(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_fw.dump")
                print("\nAn error occured in forward. Please forward snapshot_fw.dump for debugging.")
                raise ex
        else:
            num_rendered, color, rendered_likelihoods, render_velocity, render_acceleration, render_jerk, render_snap, likelihoods, radii, geomBuffer, binningBuffer, imgBuffer = _C.rasterize_graphixs(*args)

        # Keep relevant tensors for backward
        ctx.raster_settings = raster_settings
        ctx.num_rendered = num_rendered
        ctx.save_for_backward(colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, opacities, nu_degree, time, duration, velocity, acceleration, jerk, snap)
        return color, rendered_likelihoods, render_velocity, render_acceleration, render_jerk, render_snap, likelihoods, radii

    @staticmethod
    def backward(ctx, grad_out_color, _0, _1, _2, _3, _4, grad_out_likelihoods, _5):

        # Restore necessary values from context
        num_rendered = ctx.num_rendered
        raster_settings = ctx.raster_settings
        colors_precomp, means3D, scales, rotations, cov3Ds_precomp, radii, sh, geomBuffer, binningBuffer, imgBuffer, opacities, nu_degree, time, duration, velocity, acceleration, jerk, snap = ctx.saved_tensors

        # Restructure args as C++ method expects them
        args = (raster_settings.bg,
                means3D, 
                radii, 
                colors_precomp,
                opacities,
                scales, 
                rotations, 
                raster_settings.scale_modifier, 
                cov3Ds_precomp, 
                raster_settings.viewmatrix, 
                raster_settings.projmatrix, 
                raster_settings.tanfovx, 
                raster_settings.tanfovy, 
                grad_out_color, 
                grad_out_likelihoods,
                sh, 
                raster_settings.sh_degree, 
                raster_settings.campos,
                geomBuffer,
                num_rendered,
                binningBuffer,
                imgBuffer,
                raster_settings.debug,
                nu_degree,
                time,
                duration,
                velocity,
                acceleration,
                jerk,
                snap,
                raster_settings.camera_timestamp)

        # Compute gradients for relevant tensors by invoking backward method
        if raster_settings.debug:
            cpu_args = cpu_deep_copy_tuple(args) # Copy them before they can be corrupted
            try:
                grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_nu_degree, grad_time, grad_duration, grad_velocity, grad_acceleration, grad_jerk, grad_snap = _C.rasterize_graphixs_backward(*args)
            except Exception as ex:
                torch.save(cpu_args, "snapshot_bw.dump")
                print("\nAn error occured in backward. Writing snapshot_bw.dump for debugging.\n")
                raise ex
        else:
             grad_means2D, grad_colors_precomp, grad_opacities, grad_means3D, grad_cov3Ds_precomp, grad_sh, grad_scales, grad_rotations, grad_nu_degree, grad_time, grad_duration, grad_velocity, grad_acceleration, grad_jerk, grad_snap = _C.rasterize_graphixs_backward(*args)

        if torch.isnan(grad_means3D).any():
            grad_means3D[grad_means3D.isnan()] = 0.0
        if torch.isnan(grad_means2D).any():
            grad_means2D[grad_means2D.isnan()] = 0.0
        if torch.isnan(grad_sh).any():
            grad_sh[grad_sh.isnan()] = 0.0
        if torch.isnan(grad_colors_precomp).any():
            grad_colors_precomp[grad_colors_precomp.isnan()] = 0.0
        if torch.isnan(grad_opacities).any():
            grad_opacities[grad_opacities.isnan()] = 0.0
        if torch.isnan(grad_scales).any():
            grad_scales[grad_scales.isnan()] = 0.0
        if torch.isnan(grad_rotations).any():
            grad_rotations[grad_rotations.isnan()] = 0.0
        if torch.isnan(grad_cov3Ds_precomp).any():
            grad_cov3Ds_precomp[grad_cov3Ds_precomp.isnan()] = 0.0
        if torch.isnan(grad_nu_degree).any():
            grad_nu_degree[grad_nu_degree.isnan()] = 0.0
        if torch.isnan(grad_time).any():
            grad_time[grad_time.isnan()] = 0.0
        if torch.isnan(grad_duration).any():
            grad_duration[grad_duration.isnan()] = 0.0
        if torch.isnan(grad_velocity).any():
            grad_velocity[grad_velocity.isnan()] = 0.0
        if torch.isnan(grad_acceleration).any():
            grad_acceleration[grad_acceleration.isnan()] = 0.0
        if torch.isnan(grad_jerk).any():
            grad_jerk[grad_jerk.isnan()] = 0.0
        if torch.isnan(grad_snap).any():
            grad_snap[grad_snap.isnan()] = 0.0

        grads = (
            grad_means3D,
            grad_means2D,
            grad_sh,
            grad_colors_precomp,
            None,
            grad_opacities,
            grad_scales,
            grad_rotations,
            grad_cov3Ds_precomp,
            None,
            grad_nu_degree,
            grad_time,
            grad_duration,
            grad_velocity,
            grad_acceleration,
            grad_jerk,
            grad_snap,
            None,
            None
        )

        return grads

class GraphixsRasterizationSettings(NamedTuple):
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool
    camera_timestamp : float = 0.0
    inference_only : bool = False

class GraphixsRasterizer(nn.Module):
    def __init__(self, raster_settings):
        super().__init__()
        self.raster_settings = raster_settings

    def forward(self, means3D, means2D, opacities, nu_degree, time, duration, velocity, acceleration, jerk, snap, shs, scales, rotations, prev_likelihoods = None, noise_std = 0.0):

        raster_settings = self.raster_settings
        
        colors_precomp = torch.Tensor([])
        cov3D_precomp = torch.Tensor([])

        # Invoke C++/CUDA rasterization routine
        return rasterize_graphixs(
            means3D,
            means2D,
            shs,
            colors_precomp,
            prev_likelihoods,
            opacities,
            scales, 
            rotations,
            cov3D_precomp,
            raster_settings,
            nu_degree,
            time,
            duration,
            velocity,
            acceleration,
            jerk,
            snap,
            noise_std
        )

def compute_relocation_graphixs(opacity_old, scale_old, nu_degree, N, binoms, n_max):
    new_opacity, new_scale = _C.compute_relocation_graphixs(opacity_old, scale_old, nu_degree, N.int(), binoms, n_max)
    return new_opacity, new_scale