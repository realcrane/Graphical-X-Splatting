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

#include "spatial.h"
#include "simple_knn.h"

torch::Tensor
distCUDA2(const torch::Tensor& points)
{
  const int P = points.size(0);

  auto float_opts = points.options().dtype(torch::kFloat32);
  torch::Tensor means = torch::full({P}, 0.0, float_opts);
  
  SimpleKNN::knn(P, (float3*)points.contiguous().data<float>(), means.contiguous().data<float>());

  return means;
}