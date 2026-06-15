/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This file is a derivative work of the original software.
 * Modifications and additions:
 * 2026, Doga Yilmaz (doga.yilmaz@ucl.ac.uk)
 * Virtual Environments and Computer Graphics Lab, UCL
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For original inquiries contact  george.drettakis@inria.fr
 * For modification inquiries contact doga.yilmaz@ucl.ac.uk
 */

#ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
#define CUDA_RASTERIZER_BACKWARD_H_INCLUDED

#include <cuda.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#define GLM_FORCE_CUDA
#include <glm/glm.hpp>

namespace BACKWARD
{
	void render(
		const dim3 grid, dim3 block,
		const uint2* ranges,
		const uint32_t* point_list,
		int W, int H,
		const float* bg_color,
		const float2* means2D,
		const float4* conic_opacity,
		const float* colors,
		const float* final_Ts,
		const uint32_t* n_contrib,
		const float* dL_dpixels,
		const float* dL_dout_likelihoods,
		float3* dL_dmean2D,
		float4* dL_dconic2D,
		float* dL_dopacity,
		float* dL_dcolors,
		const float* nu_degree,
		float* dL_dnudegree);

	void preprocess(
		int P, int D, int M,
		const float3* means,
		const int* radii,
		const float* shs,
		const bool* clamped,
		const glm::vec3* scales,
		const glm::vec4* rotations,
		const float scale_modifier,
		const float* cov3Ds,
		const float* view,
		const float* proj,
		const float focal_x, float focal_y,
		const float tan_fovx, float tan_fovy,
		const glm::vec3* campos,
		const float3* dL_dmean2D,
		const float* dL_dconics,
		glm::vec3* dL_dmeans,
		float* dL_dcolor,
		float* dL_dcov3D,
		float* dL_dsh,
		glm::vec3* dL_dscale,
		glm::vec4* dL_drot,
		const float* opacities,
		const float* time,
		const float* duration,
		const float* velocity,
		const float* acceleration,
		const float* jerk,
		const float* snap,
		const float camera_timestamp,
		const float3* dyn_noise,
		float* dL_dopacity,
		float* dL_dtime,
		float* dL_dduration,
		float* dL_dvelocity,
		float* dL_dacceleration,
		float* dL_djerk,
		float* dL_dsnap);
}

#endif