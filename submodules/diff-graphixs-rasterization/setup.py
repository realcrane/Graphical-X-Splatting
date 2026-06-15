import os
import torch
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

# Ensure TMPDIR is set to a writable directory within the project root
project_root = os.path.dirname(os.path.abspath(__file__))
tmp_dir = os.path.join(project_root, "tmp")
os.makedirs(tmp_dir, exist_ok=True)
os.environ["TMPDIR"] = tmp_dir  # For Linux/macOS
os.environ["TEMP"] = tmp_dir    # For Windows

# Specify the GPU architecture
gpu_arch = f"{torch.cuda.get_device_capability(0)[0]}0"

setup(
    name="diff_graphixs_rasterization",
    packages=['diff_graphixs_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_graphixs_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "cuda_rasterizer/utils.cu",
                "rasterize_points.cu",
                "ext.cpp"
            ],
            extra_compile_args={
                "nvcc": [
                    "-Xcompiler", "-fno-gnu-unique",
                    "-I" + os.path.join(project_root, "third_party/glm/"),
                    f"-gencode=arch=compute_{gpu_arch},code=sm_{gpu_arch}"
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)