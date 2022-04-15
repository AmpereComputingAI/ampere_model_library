import sys
import subprocess

native_frameworks = list()

# We skip installation of given framework if an Ampere-optimized version is already present. If it is not, then we install the vanilla version (because we don't supply installation packages yet - go to https://solutions.amperecomputing.com/solutions/ampere-ai for optimized AI docker images from Ampere)

# ONNXRunTime
try:
    import onnxruntime as ort
    ort.AIO
except (AttributeError, ModuleNotFoundError):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "onnxruntime==1.11.0"])
    native_frameworks.append("ONNXRunTime")

# PyTorch
try:
    import torch
    torch._C._aio_profiler_print()
except (AttributeError, ModuleNotFoundError):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torch==1.11.0"])
    subprocess.check_call([sys.executable, "-m", "pip", "install", "torchvision==0.12.0"])
    native_frameworks.append("PyTorch")

# TensorFlow
try:
    import tensorflow as tf
    tf.AIO
except (AttributeError, ModuleNotFoundError):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "tensorflow-aarch64==2.8.0"])
    native_frameworks.append("TensorFlow")

len_native_frameworks = len(native_frameworks)
if len_native_frameworks > 0:
    native_frameworks_string = ""
    if len_native_frameworks > 1:
        for i in range(len_native_frameworks):
            if i+2 == len_native_frameworks:
                native_frameworks_string += f"{native_frameworks[i]} and "
            elif i+1 == len_native_frameworks:
                native_frameworks_string += f"{native_frameworks[i]}"
            else:
                native_frameworks_string += f"{native_frameworks[i]}, "
    print(f"\033[1;31m \nCAUTION: {native_frameworks_string} frameworks have been installed in their native versions missing Ampere optimizations. Consider using AI-dedicated Docker images for increased performance. Available at: https://solutions.amperecomputing.com/solutions/ampere-ai")
