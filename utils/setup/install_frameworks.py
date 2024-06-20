# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import sys
import subprocess

native_frameworks = list()

# We skip installation of given framework if an Ampere-optimized version is already present. If it is not, then we
# install the vanilla version (because we don't supply installation packages yet - go to
# https://solutions.amperecomputing.com/solutions/ampere-ai for optimized AI docker images from Ampere)

# TensorFlow
try:
    import tensorflow  # noqa
except ModuleNotFoundError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow"])
    except subprocess.CalledProcessError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "tensorflow"])
    native_frameworks.append("TensorFlow")

# ONNXRunTime
try:
    import onnxruntime  # noqa
except ModuleNotFoundError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime"])
    except subprocess.CalledProcessError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "onnxruntime"])
    native_frameworks.append("ONNXRunTime")

# PyTorch
try:
    import torch  # noqa
except ModuleNotFoundError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torch"])
    except subprocess.CalledProcessError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "torch"])
    native_frameworks.append("PyTorch")
try:
    import torchvision  # noqa
except ModuleNotFoundError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision"])
    except subprocess.CalledProcessError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "--break-system-packages", "torchvision"])
# summary
len_native_frameworks = len(native_frameworks)
if len_native_frameworks > 0:
    native_frameworks_string = ""
    for i in range(len_native_frameworks):
        if i + 2 == len_native_frameworks:
            native_frameworks_string += f"{native_frameworks[i]} and "
        elif i + 1 == len_native_frameworks:
            native_frameworks_string += f"{native_frameworks[i]}"
        else:
            native_frameworks_string += f"{native_frameworks[i]}, "
    print(f"\033[1;31m \nCAUTION: {native_frameworks_string} frameworks have been installed in their native versions "
          f"missing Ampere optimizations. Consider using AI-dedicated Docker images for increased performance. "
          f"Available at: https://solutions.amperecomputing.com/solutions/ampere-ai")
