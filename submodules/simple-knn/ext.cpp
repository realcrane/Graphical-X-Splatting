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

#include <torch/extension.h>
#include "spatial.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("distCUDA2", &distCUDA2);
}
