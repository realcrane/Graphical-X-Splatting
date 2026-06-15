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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include "cuda_rasterizer/utils.h"
#include <fstream>
#include <string>
#include <functional>

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizeGraphixsCUDA(
	const torch::Tensor& background,
	const torch::Tensor& means3D,
    const torch::Tensor& colors,
	const torch::Tensor& prev_likelihoods,
    const torch::Tensor& opacity,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const torch::Tensor& nu_degree,
	const torch::Tensor& time,
	const torch::Tensor& duration,
	const torch::Tensor& velocity,
	const torch::Tensor& acceleration,
	const torch::Tensor& jerk,
	const torch::Tensor& snap,
	const float camera_timestamp,
	const float noise_std,
	const bool inference_only)
{
  if (means3D.ndimension() != 2 || means3D.size(1) != 3) {
    AT_ERROR("means3D must have dimensions (num_points, 3)");
  }
  
  const int P = means3D.size(0);
  const int H = image_height;
  const int W = image_width;

  auto int_opts = means3D.options().dtype(torch::kInt32);
  auto float_opts = means3D.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor out_rendered_likelihoods = inference_only ? torch::empty({0}, float_opts) : torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_velocity = inference_only ? torch::empty({0}, float_opts) : torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_acceleration = inference_only ? torch::empty({0}, float_opts) : torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_jerk = inference_only ? torch::empty({0}, float_opts) : torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_snap = inference_only ? torch::empty({0}, float_opts) : torch::full({1, H, W}, 0.0, float_opts);
  torch::Tensor out_likelihoods = inference_only ? torch::empty({0}, float_opts) : torch::full({P}, 0.0, float_opts);

  torch::Tensor radii = torch::full({P}, 0, means3D.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, degree, M,
		background.contiguous().data<float>(),
		W, H,
		means3D.contiguous().data<float>(),
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data<float>(), 
		prev_likelihoods.contiguous().data<float>(),
		opacity.contiguous().data<float>(), 
		scales.contiguous().data_ptr<float>(),
		scale_modifier,
		rotations.contiguous().data_ptr<float>(),
		cov3D_precomp.contiguous().data<float>(), 
		viewmatrix.contiguous().data<float>(), 
		projmatrix.contiguous().data<float>(),
		campos.contiguous().data<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data<float>(),
		out_rendered_likelihoods.contiguous().data<float>(),
		out_velocity.contiguous().data<float>(),
		out_acceleration.contiguous().data<float>(),
		out_jerk.contiguous().data<float>(),
		out_snap.contiguous().data<float>(),
		out_likelihoods.contiguous().data<float>(),
		nu_degree.contiguous().data<float>(),
		time.contiguous().data<float>(),
		duration.contiguous().data<float>(),
		velocity.contiguous().data<float>(),
		acceleration.contiguous().data<float>(),
		jerk.contiguous().data<float>(),
		snap.contiguous().data<float>(),
		camera_timestamp,
		noise_std,
		radii.contiguous().data<int>(),
		debug,
		inference_only);
  }
  return std::make_tuple(rendered, out_color, out_rendered_likelihoods, out_velocity, out_acceleration, out_jerk, out_snap, out_likelihoods, radii, geomBuffer, binningBuffer, imgBuffer);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizeGraphixsBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& means3D,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& opacities,
	const torch::Tensor& scales,
	const torch::Tensor& rotations,
	const float scale_modifier,
	const torch::Tensor& cov3D_precomp,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_likelihoods,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug,
	const torch::Tensor& nu_degree,
	const torch::Tensor& time,
	const torch::Tensor& duration,
	const torch::Tensor& velocity,
	const torch::Tensor& acceleration,
	const torch::Tensor& jerk,
	const torch::Tensor& snap,
	const float camera_timestamp) 
{
  const int P = means3D.size(0);
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  torch::Tensor dL_dmeans3D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dcolors = torch::zeros({P, NUM_CHANNELS}, means3D.options());
  torch::Tensor dL_dconic = torch::zeros({P, 2, 2}, means3D.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dcov3D = torch::zeros({P, 6}, means3D.options());
  torch::Tensor dL_dsh = torch::zeros({P, M, 3}, means3D.options());
  torch::Tensor dL_dscales = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_drotations = torch::zeros({P, 4}, means3D.options());

  // gradient for nu_degree
  torch::Tensor dL_dnudegree = torch::zeros({P, 1}, means3D.options());

  // gradient for time, duration, velocity, acceleration
  torch::Tensor dL_dtime = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dduration = torch::zeros({P, 1}, means3D.options());
  torch::Tensor dL_dvelocity = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dacceleration = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_djerk = torch::zeros({P, 3}, means3D.options());
  torch::Tensor dL_dsnap = torch::zeros({P, 3}, means3D.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, degree, M, R,
	  background.contiguous().data<float>(),
	  W, H, 
	  means3D.contiguous().data<float>(),
	  sh.contiguous().data<float>(),
	  colors.contiguous().data<float>(),
	  opacities.contiguous().data<float>(),
	  scales.data_ptr<float>(),
	  scale_modifier,
	  rotations.data_ptr<float>(),
	  cov3D_precomp.contiguous().data<float>(),
	  viewmatrix.contiguous().data<float>(),
	  projmatrix.contiguous().data<float>(),
	  campos.contiguous().data<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data<float>(),
	  dL_dout_likelihoods.contiguous().data<float>(),
	  dL_dmeans2D.contiguous().data<float>(),
	  dL_dconic.contiguous().data<float>(),  
	  dL_dopacity.contiguous().data<float>(),
	  dL_dcolors.contiguous().data<float>(),
	  dL_dmeans3D.contiguous().data<float>(),
	  dL_dcov3D.contiguous().data<float>(),
	  dL_dsh.contiguous().data<float>(),
	  dL_dscales.contiguous().data<float>(),
	  dL_drotations.contiguous().data<float>(),
	  debug,
	  nu_degree.contiguous().data<float>(),
	  dL_dnudegree.contiguous().data<float>(),
	  time.contiguous().data<float>(),
	  duration.contiguous().data<float>(),
	  velocity.contiguous().data<float>(),
	  acceleration.contiguous().data<float>(),
	  jerk.contiguous().data<float>(),
	  snap.contiguous().data<float>(),
	  camera_timestamp,
	  dL_dtime.contiguous().data<float>(),
	  dL_dduration.contiguous().data<float>(),
	  dL_dvelocity.contiguous().data<float>(),
	  dL_dacceleration.contiguous().data<float>(),
	  dL_djerk.contiguous().data<float>(),
	  dL_dsnap.contiguous().data<float>());
  }

  return std::make_tuple(dL_dmeans2D, dL_dcolors, dL_dopacity, dL_dmeans3D, dL_dcov3D, dL_dsh, dL_dscales, dL_drotations, dL_dnudegree, dL_dtime, dL_dduration, dL_dvelocity, dL_dacceleration, dL_djerk, dL_dsnap);
}

std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationGraphixsCUDA(
	torch::Tensor& opacity_old,
	torch::Tensor& scale_old,
	torch::Tensor& nu_degree,
	torch::Tensor& N,
	torch::Tensor& binoms,
	const int n_max)
{
	const int P = opacity_old.size(0);
  
	torch::Tensor final_opacity = torch::full({P}, 0, opacity_old.options().dtype(torch::kFloat32));
	torch::Tensor final_scale = torch::full({3 * P}, 0, scale_old.options().dtype(torch::kFloat32));

	if(P != 0)
	{
		UTILS::ComputeRelocationGraphixs(P,
			opacity_old.contiguous().data<float>(),
			scale_old.contiguous().data<float>(),
			nu_degree.contiguous().data<float>(),
			N.contiguous().data<int>(),
			binoms.contiguous().data<float>(),
			n_max,
			final_opacity.contiguous().data<float>(),
			final_scale.contiguous().data<float>());
	}

	return std::make_tuple(final_opacity, final_scale);

}