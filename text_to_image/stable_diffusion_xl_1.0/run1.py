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
n_steps = 40
prompt = "An astronaut riding a green horse"

#with torch.no_grad():
#   images = model(prompt=prompt).images[0]


image = model(
    prompt=prompt,
    num_inference_steps=n_steps,
    denoising_end=0.8,
    output_type="latent",
).images

print(type(image))

