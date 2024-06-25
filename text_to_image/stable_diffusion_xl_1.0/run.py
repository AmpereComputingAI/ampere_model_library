from diffusers import DiffusionPipeline, StableDiffusionPipeline, StableDiffusionXLPipeline
import torch
from utils.pytorch import PyTorchRunnerV2, apply_compile
import os

torch.set_num_threads(128)

model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True)

model.to("cpu")
model.unet = apply_compile(model.unet)
# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "An astronaut riding a green horse"

with torch.no_grad():
    images = model(prompt=prompt).images[0]