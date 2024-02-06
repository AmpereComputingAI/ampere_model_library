# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import os
import sys
import subprocess
from packaging import version

native_frameworks = list()

# We skip installation of given framework if an Ampere-optimized version is already present. If it is not, then we
# install the vanilla version (because we don't supply installation packages yet - go to
# https://solutions.amperecomputing.com/solutions/ampere-ai for optimized AI docker images from Ampere)

# TensorFlow
try:
    import tensorflow as tf
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"tensorflow"])
    native_frameworks.append("TensorFlow")

# ONNXRunTime
try:
    import onnxruntime as ort
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"onnxruntime"])
    native_frameworks.append("ONNXRunTime")

# PyTorch
try:
    import torch
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"torch<2.2"])
    native_frameworks.append("PyTorch")
try:
    import torchvision
except ModuleNotFoundError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"torchvision==0.16.2"])


# summary
len_native_frameworks = len(native_frameworks)
if len_native_frameworks > 0:
    native_frameworks_string = ""
    for i in range(len_native_frameworks):
            if i+2 == len_native_frameworks:
                native_frameworks_string += f"{native_frameworks[i]} and "
            elif i+1 == len_native_frameworks:
                native_frameworks_string += f"{native_frameworks[i]}"
            else:
                native_frameworks_string += f"{native_frameworks[i]}, "      
    print(f"\033[1;31m \nCAUTION: {native_frameworks_string} frameworks have been installed in their native versions "
          f"missing Ampere optimizations. Consider using AI-dedicated Docker images for increased performance. "
          f"Available at: https://solutions.amperecomputing.com/solutions/ampere-ai")
