from diffusers import DiffusionPipeline
import torch

torch.set_num_threads(128)

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
pipe.to("cpu")

# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "An astronaut riding a green horse"

with torch.no_grad():
    images = pipe(prompt=prompt).images[0]
