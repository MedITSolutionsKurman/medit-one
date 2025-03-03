#!/usr/bin/env python3
"""
Custom installation script to explicitly build CUDA extensions for MedIT ONE.
This script should be run directly after installing the package.

Usage:
    pip install -e .
    python install_cuda.py

Or for a one-line command:
    pip install -e . && python install_cuda.py
"""

import os
import sys
import subprocess
import importlib.util

print("\n==================================================================")
print("MedIT-ONE CUDA EXTENSION INSTALLER")
print("==================================================================\n")

# Check if torch is installed
try:
    import torch

    print(f"PyTorch version: {torch.__version__}")
    if not torch.cuda.is_available():
        print(
            "CUDA is not available in PyTorch. Please install a CUDA-enabled PyTorch version."
        )
        sys.exit(1)

    print(f"CUDA is available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device count: {torch.cuda.device_count()}")

except ImportError:
    print(
        "PyTorch is not installed. Please install a CUDA-enabled PyTorch version first."
    )
    sys.exit(1)

# Check if the source files exist
current_dir = os.path.dirname(os.path.abspath(__file__))
cpp_path = os.path.join(current_dir, "csrc", "one_kernels.cpp")
cu_path = os.path.join(current_dir, "csrc", "flash_attention_kernel.cu")

if not os.path.exists(cpp_path) or not os.path.exists(cu_path):
    print(f"Source files not found:")
    print(f"  - {cpp_path}: {os.path.exists(cpp_path)}")
    print(f"  - {cu_path}: {os.path.exists(cu_path)}")
    print("CUDA extension cannot be built.")
    sys.exit(1)

print("Source files found. Building CUDA extension...")

# Check if cpp_extension is available
try:
    from torch.utils import cpp_extension
except ImportError:
    print(
        "torch.utils.cpp_extension not found. Make sure you have a complete PyTorch installation."
    )
    sys.exit(1)

# Build the CUDA extension directly
try:
    print("Building one_turbo extension...")

    # Set build directory
    build_dir = os.path.join(current_dir, "build")
    os.makedirs(build_dir, exist_ok=True)

    # Build the extension
    ext = cpp_extension.load(
        name="one_turbo",
        sources=[cpp_path, cu_path],
        build_directory=build_dir,
        extra_cflags=["-O3", "-fopenmp"],
        extra_cuda_cflags=[
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
        verbose=True,
    )

    print("\nCUDA extension successfully built!")
    print(f"Module location: {ext.__file__}")

    # Create a file to indicate successful installation
    with open(os.path.join(current_dir, "one", "cuda_installed"), "w") as f:
        f.write("CUDA extension successfully installed on " + torch.version.cuda)

    print("\nTo verify the installation, run:")
    print(
        "python -c \"from one.turbo_ops import TURBO_MODE; print(f'CUDA Turbo Mode: {\\'Enabled\\' if TURBO_MODE else \\'Disabled\\'}')\""
    )

except Exception as e:
    print(f"Error building CUDA extension: {e}")
    sys.exit(1)
