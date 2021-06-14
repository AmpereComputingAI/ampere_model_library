import os
from utils.tf import TFFrozenModelRunner

img_width = 128
img_height = 128
img_channels = 3

unet = TFFrozenModelRunner(path_to_model='/onspecta/dev/model_zoo/models/unet/unet.pb', output_names=['Identity'])

