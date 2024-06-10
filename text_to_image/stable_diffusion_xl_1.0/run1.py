from diffusers import DiffusionPipeline, StableDiffusionPipeline
import torch

from utils.pytorch import apply_jit_trace

torch.set_num_threads(128)

pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                         torch_dtype=torch.float16,
                                         use_safetensors=True,
                                         variant="fp16")

pipe.to("cpu")
unet = pipe.unet
unet.eval()

example_input = torch.randn(1, 3, 512, 512)
# traced_model = torch.jit.trace(pipe, example_input)

# frozen_model = torch.jit.freeze(traced_model)

model = apply_jit_trace(unet, example_input)
# if using torch < 2.0
# pipe.enable_xformers_memory_efficient_attention()

prompt = "An astronaut riding a green horse"

with torch.no_grad():
    images = model(prompt=prompt).images[0]
