from diffusers import DiffusionPipeline
import torch
from utils.pytorch import apply_compile

torch.set_num_threads(128)

model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", use_safetensors=True)
model.to("cpu")

prompt = "An astronaut riding a green horse"

with torch.no_grad():
    images = model(prompt=prompt).images[0]
