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


# ONNXRunTime
native_ort_version = "1.11.0"

ampere_ort = False
install_ort = False

try:
    import onnxruntime as ort
    if version.parse(ort.__version__) != version.parse(native_ort_version):
        install_ort = True
    ort.AIO
    ampere_ort = True
except ModuleNotFoundError:
    install_ort = True
except AttributeError:
    ampere_ort = False

if not ampere_ort and install_ort:
    subprocess.check_call([sys.executable, "-m", "pip", "install", f"onnxruntime=={native_ort_version}"])
    native_frameworks.append("ONNXRunTime")


# PyTorch
native_torch_version = "1.11.0"
torchvision_version = "0.12.0"

ampere_torch = False
install_torch = False
install_torchvision = False

try:
    import torch
    if version.parse(torch.__version__) != version.parse(native_torch_version):
        install_torch = True
    torch._C._aio_profiler_print()
    ampere_torch = True
except ModuleNotFoundError:
    install_torch = True
except AttributeError:
    ampere_torch = False

try:
    import torchvision
    if version.parse(torchvision.__version__) != version.parse(torchvision_version):
        install_torchvision = True
except ModuleNotFoundError:
    install_torchvision = True

if not ampere_torch:
    if install_torch:
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"torch=={native_torch_version}"])
    if install_torchvision:
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"torchvision=={torchvision_version}"])
    native_frameworks.append("PyTorch")


# TensorFlow
native_tf_version = "2.8.0"

ampere_tf = False
install_tf = False

try:
    import tensorflow as tf
    if version.parse(tf.__version__) != version.parse(native_tf_version):
        install_tf = True
    tf.AIO
    ampere_tf = True
except ModuleNotFoundError:
    install_tf = True
except AttributeError:
    ampere_tf = False

if not ampere_tf and install_tf:
    if os.environ["ARCH"] == "aarch64":
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"tensorflow-aarch64=={native_tf_version}"])
    else:
        subprocess.check_call([sys.executable, "-m", "pip", "install", f"tensorflow=={native_tf_version}"])
    native_frameworks.append("TensorFlow")


# summary
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
    print(f"\033[1;31m \nCAUTION: {native_frameworks_string} frameworks have been installed in their native versions "
          f"missing Ampere optimizations. Consider using AI-dedicated Docker images for increased performance. "
          f"Available at: https://solutions.amperecomputing.com/solutions/ampere-ai")
