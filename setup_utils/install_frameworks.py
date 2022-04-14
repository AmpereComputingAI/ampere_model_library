from setuptools.command.easy_install import main as install

# We skip installation if Ampere-optimized framework is present already

# ONNXRunTime
try:
    import onnxruntime as ort
    print(ort.__version__)
    ort.AIO
except (AttributeError, ModuleNotFoundError):
    install(["onnxruntime==1.11.0"])

# PyTorch
try:
    import torch
    print(torch.__version__)
    torch._C._aio_profiler_print()
except (AttributeError, ModuleNotFoundError):
    install(["torch==1.11.0"])

# TensorFlow
try:
    import tensorflow as tf
    print(tf.__version__)
    tf.AIO
except (AttributeError, ModuleNotFoundError):
    install(["tensorflow-aarch64==2.8.0"])
