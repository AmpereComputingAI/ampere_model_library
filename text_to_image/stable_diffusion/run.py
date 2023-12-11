import os
import sys

import torch
import numpy as np
from PIL import Image
from pathlib import Path
from einops import rearrange


def run_pytorch_fp32(model_path, config, steps, scale, batch_size, num_runs, timeout):
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "stablediffusion"))

    from omegaconf import OmegaConf
    from contextlib import nullcontext
    from utils.benchmark import run_model
    from utils.pytorch import PyTorchRunnerV2
    from pytorch_lightning import seed_everything
    from utils.text_to_image.stable_diffusion import StableDiffusion
    from text_to_image.stable_diffusion.stablediffusion.ldm.models.diffusion.ddim import DDIMSampler
    from text_to_image.stable_diffusion.stablediffusion.scripts.txt2img import load_model_from_config

    seed_everything(42)
    config = OmegaConf.load(f"{config}")
    model = load_model_from_config(config, f"{model_path}", torch.device("cpu"))
    sampler = DDIMSampler(model, device=torch.device("cpu"))
    shape = [4, 512 // 8, 512 // 8]

    unet = model.model.diffusion_model
    decoder = model.first_stage_model.decoder
    stablediffusion_data = Path(os.path.dirname(os.path.abspath(__file__)), 'models')
    unet_path = Path(stablediffusion_data, "unet.pt")
    decoder_path = Path(stablediffusion_data, "decoder.pt")
    if not stablediffusion_data.exists():
        stablediffusion_data.mkdir(exist_ok=True)

    with torch.no_grad(), nullcontext():
        if unet_path.exists():
            scripted_unet = torch.jit.load(unet_path)
        else:
            img_in = torch.ones(2, 4, 96, 96, dtype=torch.float32)
            t_in = torch.ones(2, dtype=torch.int64)
            context = torch.ones(2, 77, 1024, dtype=torch.float32)
            scripted_unet = torch.jit.trace(unet, (img_in, t_in, context))
            scripted_unet = torch.jit.freeze(scripted_unet)
            torch.jit.save(scripted_unet, unet_path)
        model.model.scripted_diffusion_model = scripted_unet

        if decoder_path.exists():
            scripted_decoder = torch.jit.load(decoder_path)
        else:
            samples_ddim = torch.ones(1, 4, 96, 96, dtype=torch.float32)
            scripted_decoder = torch.jit.trace(decoder, (samples_ddim))
            scripted_decoder = torch.jit.freeze(scripted_decoder)
            torch.jit.save(scripted_decoder, decoder_path)
        model.first_stage_model.decoder = scripted_decoder

    def single_pass_pytorch(_runner, _stablediffusion):
        prompt = _stablediffusion.get_input()
        x_samples = _runner.run(batch_size * steps, prompt=prompt)
        _stablediffusion.submit_count(batch_size, x_samples)

    def wrapper(prompt):
        samples, _ = sampler.sample(S=steps,
                                    conditioning=model.get_learned_conditioning([prompt] * batch_size),
                                    batch_size=batch_size,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=scale,
                                    unconditional_conditioning=model.get_learned_conditioning(batch_size * [""]) if scale != 1.0 else None,
                                    eta=0.0,
                                    x_T=None)
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        return x_samples

    runner = PyTorchRunnerV2(wrapper)
    stablediffusion = StableDiffusion()

    return run_model(single_pass_pytorch, runner, stablediffusion, batch_size, num_runs, timeout)


def run_pytorch_cuda(model_path, config, steps, scale, batch_size, num_runs, timeout):
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "stablediffusion"))

    from omegaconf import OmegaConf
    from contextlib import nullcontext
    from utils.benchmark import run_model
    from utils.pytorch import PyTorchRunnerV2
    from pytorch_lightning import seed_everything
    from utils.text_to_image.stable_diffusion import StableDiffusion
    from text_to_image.stable_diffusion.stablediffusion.ldm.models.diffusion.ddim import DDIMSampler
    from text_to_image.stable_diffusion.stablediffusion.scripts.txt2img import load_model_from_config

    seed_everything(42)
    config = OmegaConf.load(f"{config}")
    model = load_model_from_config(config, f"{model_path}", torch.device("cuda"))
    sampler = DDIMSampler(model, device=torch.device("cuda"))
    shape = [4, 512 // 8, 512 // 8]

    stablediffusion_data = Path(os.path.dirname(os.path.abspath(__file__)), 'models')

    if not stablediffusion_data.exists():
        stablediffusion_data.mkdir(exist_ok=True)

    def single_pass_pytorch(_runner, _stablediffusion):
        prompt = _stablediffusion.get_input()
        x_samples = _runner.run(batch_size * steps, prompt=prompt)
        _stablediffusion.submit_count(batch_size, x_samples)

    def wrapper(prompt):
        samples, _ = sampler.sample(S=steps,
                                    conditioning=model.get_learned_conditioning([prompt] * batch_size),
                                    batch_size=batch_size,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=scale,
                                    unconditional_conditioning=model.get_learned_conditioning(batch_size * [""]) if scale != 1.0 else None,
                                    eta=0.0,
                                    x_T=None)
        x_samples = model.decode_first_stage(samples)
        x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        return x_samples

    runner = PyTorchRunnerV2(wrapper)
    stablediffusion = StableDiffusion()

    return run_model(single_pass_pytorch, runner, stablediffusion, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser

    parser = DefaultArgParser(["pytorch"])
    parser.ask_for_batch_size()
    parser.require_model_path()
    parser.add_argument("--config", type=str, required=True,
                        help="path to config which constructs model")
    parser.add_argument("--steps", type=int, default=25, help="steps through which the model processes the input")
    parser.add_argument('--scale', type=int, default=9, help="scale of the image")

    import torch
    if torch.cuda.is_available():
        run_pytorch_cuda(**vars(parser.parse()))
    else:
        run_pytorch_fp32(**vars(parser.parse()))
