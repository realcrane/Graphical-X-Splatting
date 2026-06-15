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

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import torch

# Ensure TMPDIR is set to a writable directory within the project root
project_root = os.path.dirname(os.path.abspath(__file__))
tmp_dir = os.path.join(project_root, "tmp")
os.makedirs(tmp_dir, exist_ok=True)
os.environ["TMPDIR"] = tmp_dir  # For Linux/macOS
os.environ["TEMP"] = tmp_dir    # For Windows

cxx_compiler_flags = []

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

gpu_arch = f"{torch.cuda.get_device_capability(0)[0]}0"

setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
                "spatial.cu", 
                "simple_knn.cu",
                "ext.cpp"
            ],
            extra_compile_args={
                "nvcc": [f"-gencode=arch=compute_{gpu_arch},code=sm_{gpu_arch}"],
                "cxx": cxx_compiler_flags
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)