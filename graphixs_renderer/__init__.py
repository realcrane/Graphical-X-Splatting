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
import math
from diff_graphixs_rasterization import GraphixsRasterizationSettings, GraphixsRasterizer
from scene.nt_model import NTModel


def render(viewpoint_camera, pc : NTModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, dynamics_noise_std=0.0, inference_only=False):
    """
    Render the scene.
    Background tensor (bg_color) must be on GPU!
    """
    rotations = pc.get_rotation
    scales = pc.get_scaling
    opacity = pc.get_base_opacity
    shs = pc.get_features
    time = pc.get_origin_time
    duration = pc.get_duration
    means3D = pc.get_xyz
    velocity = pc.get_velocity
    acceleration = pc.get_acceleration
    jerk = pc.get_jerk
    snap = pc.get_snap

    if inference_only:
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, device="cuda")
    else:
        screenspace_points = torch.zeros_like(means3D, dtype=means3D.dtype, requires_grad=True, device="cuda") + 0
        try:
            screenspace_points.retain_grad()
        except:
            pass

    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    raster_settings = GraphixsRasterizationSettings(
        image_height=int(viewpoint_camera.image_height),
        image_width=int(viewpoint_camera.image_width),
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug,
        camera_timestamp=viewpoint_camera.timestamp,
        inference_only=inference_only
    )
    rasterizer = GraphixsRasterizer(raster_settings=raster_settings)

    means2D = screenspace_points
    prev_likelihoods = pc.get_likelihoods
    nu_degree = pc.get_nu_degree

    # Rasterize visible Gaussians to image, obtain their radii (on screen).
    rendered_image, rendered_likelihoods, rendered_velocity, rendered_acceleration, rendered_jerk, rendered_snap, likelihoods, radii = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        prev_likelihoods = prev_likelihoods,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        nu_degree = nu_degree,
        time = time,
        duration = duration,
        velocity = velocity,
        acceleration = acceleration,
        jerk = jerk,
        snap = snap,
        noise_std = dynamics_noise_std
    )

    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "rendered_likelihoods": rendered_likelihoods,
            "rendered_velocity": rendered_velocity,
            "rendered_acceleration": rendered_acceleration,
            "rendered_jerk": rendered_jerk,
            "rendered_snap": rendered_snap,
            "likelihoods": likelihoods,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii}
