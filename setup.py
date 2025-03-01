"""
Setup script for medit-one package.
This is a thin wrapper around pyproject.toml that adds CUDA extension support.
"""

from setuptools import setup, find_packages
import os
import sys

# Check if we're building with CUDA support
cuda_extensions = []

try:
    import torch
    from torch.utils import cpp_extension

    # Only add CUDA extension if torch is available and CUDA is available
    if torch.cuda.is_available():
        # Use relative paths instead of absolute paths
        cuda_extension = cpp_extension.CUDAExtension(
            name="one_turbo",
            sources=[
                os.path.join("csrc", "one_kernels.cpp"),
                os.path.join("csrc", "flash_attention_kernel.cu"),
            ],
            extra_compile_args={
                "cxx": ["-O3", "-fopenmp"],
                "nvcc": [
                    "-O3",
                    "-gencode=arch=compute_70,code=sm_70",  # Volta
                    "-gencode=arch=compute_75,code=sm_75",  # Turing
                    "-gencode=arch=compute_80,code=sm_80",  # Ampere
                    "-gencode=arch=compute_86,code=sm_86",  # Ampere (RTX 30xx)
                    "-gencode=arch=compute_89,code=sm_89",  # Hopper
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "-diag-suppress=177",  # Suppress unused variable warnings
                    "-diag-suppress=186",  # Suppress pointless comparison warnings
                    "-diag-suppress=20012",  # Suppress implicit int/float conversion warnings
                ],
            },
        )

        cuda_extensions.append(cuda_extension)
        print("CUDA is available. Building with CUDA extensions.")
    else:
        print("CUDA is not available. Building without CUDA extensions.")

except ImportError:
    print("PyTorch not found. Building without CUDA extensions.")

# Let setuptools handle the rest using pyproject.toml as the source of truth
setup(
    ext_modules=cuda_extensions,
    cmdclass={"build_ext": cpp_extension.BuildExtension} if cuda_extensions else {},
)
