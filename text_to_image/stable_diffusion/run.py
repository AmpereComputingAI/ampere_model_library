import os
import sys
import time

import torch
import numpy as np
from PIL import Image
from torch import autocast
from einops import rearrange
from tqdm import tqdm, trange
from omegaconf import OmegaConf
from torchvision.utils import make_grid
from contextlib import nullcontext

from pathlib import Path

from utils.benchmark import run_model
from utils.pytorch import PyTorchRunnerV2
# from utils.text_to_image.stable_diffusion import StableDiffusion
from text_to_image.stable_diffusion.stablediffusion.scripts.txt2img import load_model_from_config
from text_to_image.stable_diffusion.stablediffusion.ldm.models.diffusion.ddim import DDIMSampler


def run_pytorch_fp32(model_name, num_runs, timeout):
    batch_size = 1
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "stablediffusion"))

    # checkpoint_link = "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/v2-1_512-ema-pruned.ckpt"
    checkpoint = "/ampere/v2-1_512-ema-pruned.ckpt"
    config = OmegaConf.load("/ampere/aml/text_to_image/stable_diffusion/stablediffusion/configs/stable-diffusion/intel/v2-inference-fp32.yaml")
    device = torch.device("cpu")
    model = load_model_from_config(config, checkpoint, device)
    sampler = DDIMSampler(model, device=device)

    prompt = 'a professional photograph of an astronaut riding a triceratops'
    # precision_scope = autocast
    # precision_scope(device)

    # batch_size = 3
    # n_iter = 3
    batch_size = 1
    n_iter = 1
    scale = 9
    H = 512
    W = 512
    C = 4
    f = 8
    steps = 25
    n_samples = batch_size
    n_rows = batch_size
    ddim_eta = 0.0
    start_code = None
    ipex = False
    torchscript = True
    bf16 = False
    precision = "autocast"

    outpath = "outputs/txt2img-samples"
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    sample_count = 0
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    data = [batch_size * [prompt]]
    sample_time = 0

    if torchscript or ipex:
        transformer = model.cond_stage_model.model
        unet = model.model.diffusion_model
        decoder = model.first_stage_model.decoder
        additional_context = torch.cpu.amp.autocast() if bf16 else nullcontext()
        shape = [C, H // f, W // f]

        if bf16 and not torchscript and not ipex:
            raise ValueError('Bfloat16 is supported only for torchscript+ipex')
        if bf16 and unet.dtype != torch.bfloat16:
            raise ValueError("Use configs/stable-diffusion/intel/ configs with bf16 enabled if " +
                             "you'd like to use bfloat16 with CPU.")
        if unet.dtype == torch.float16 and device == torch.device("cpu"):
            raise ValueError(
                "Use configs/stable-diffusion/intel/ configs for your model if you'd like to run it on CPU.")

        if ipex:
            import intel_extension_for_pytorch as ipex
            bf16_dtype = torch.bfloat16 if bf16 else None
            transformer = transformer.to(memory_format=torch.channels_last)
            transformer = ipex.optimize(transformer, level="O1", inplace=True)

            unet = unet.to(memory_format=torch.channels_last)
            unet = ipex.optimize(unet, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

            decoder = decoder.to(memory_format=torch.channels_last)
            decoder = ipex.optimize(decoder, level="O1", auto_kernel_selection=True, inplace=True, dtype=bf16_dtype)

        if torchscript:
            with torch.no_grad(), additional_context:
                # get UNET scripted
                if unet.use_checkpoint:
                    raise ValueError("Gradient checkpoint won't work with tracing. " +
                                     "Use configs/stable-diffusion/intel/ configs for your model or disable checkpoint in your config.")

                cache_dir = Path(Path.home(), "cache_stable_diff")
                if not cache_dir.exists():
                    cache_dir.mkdir(exist_ok=True)

                unet_path = Path(cache_dir, "unet.pt")
                if unet_path.exists():
                    scripted_unet = torch.jit.load(unet_path)
                else:
                    img_in = torch.ones(2, 4, 96, 96, dtype=torch.float32)
                    t_in = torch.ones(2, dtype=torch.int64)
                    context = torch.ones(2, 77, 1024, dtype=torch.float32)
                    scripted_unet = torch.jit.trace(unet, (img_in, t_in, context))
                    scripted_unet = torch.jit.freeze(scripted_unet)
                    torch.jit.save(scripted_unet, unet_path)
                    print(unet_path)
                print(type(scripted_unet))
                model.model.scripted_diffusion_model = scripted_unet

                # get Decoder for first stage model scripted
                decoder_path = Path(cache_dir, "decoder.pt")
                if decoder_path.exists():
                    scripted_decoder = torch.jit.load(decoder_path)
                else:
                    samples_ddim = torch.ones(1, 4, 96, 96, dtype=torch.float32)
                    scripted_decoder = torch.jit.trace(decoder, (samples_ddim))
                    scripted_decoder = torch.jit.freeze(scripted_decoder)
                    torch.jit.save(scripted_decoder, decoder_path)
                    print(decoder_path)
                print(type(scripted_decoder))
                model.first_stage_model.decoder = scripted_decoder

        prompts = data[0]
        print("Running a forward pass to initialize optimizations")
        uc = None
        if scale != 1.0:
            uc = model.get_learned_conditioning(batch_size * [""])
        if isinstance(prompts, tuple):
            prompts = list(prompts)

        with torch.no_grad(), additional_context:
            for _ in range(3):
                c = model.get_learned_conditioning(prompts)
            samples_ddim, _ = sampler.sample(S=5,
                                             conditioning=c,
                                             batch_size=batch_size,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=scale,
                                             unconditional_conditioning=uc,
                                             eta=ddim_eta,
                                             x_T=start_code)
            print("Running a forward pass for decoder")
            for _ in range(3):
                x_samples_ddim = model.decode_first_stage(samples_ddim)

    precision_scope = autocast if precision == "autocast" or bf16 else nullcontext
    with torch.no_grad(), precision_scope(str(device)), model.ema_scope():
    # with torch.no_grad(), model.ema_scope():
        all_samples = list()
        for n in trange(n_iter, desc="Sampling"):
            for prompts in tqdm(data, desc="data"):
                uc = None
                if scale != 1.0:
                    uc = model.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = model.get_learned_conditioning(prompts)
                shape = [C, H // f, W // f]
                start = time.time()
                samples, _ = sampler.sample(S=steps,
                                            conditioning=c,
                                            batch_size=n_samples,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            x_T=start_code)
                end = time.time()

                print(end - start)

                sample_time += (end - start)

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    # img = put_watermark(img, wm_encoder)
                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                    base_count += 1
                    sample_count += 1

                all_samples.append(x_samples)

        # additionally, save as grid
        grid = torch.stack(all_samples, 0)
        grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        grid = make_grid(grid, nrow=n_rows)

        # to image
        grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        grid = Image.fromarray(grid.astype(np.uint8))
        # grid = put_watermark(grid, wm_encoder)
        grid.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
        grid_count += 1

    print('sample time is:', sample_time)

    # def single_pass_pytorch(runner, stablediffusion):
    #     array = stablediffusion.get_input_array()
    #     stablediffusion.submit_transcription(runner.run(batch_size * array.shape[0], audio)["text"].lstrip().replace(".", "").upper())

    # runner = PyTorchRunnerV2(model)
    # stablediffusion = StableDiffusion()


    # return run_model(single_pass_pytorch, runner, stablediffusion, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser
    stable_diffusion_variants = ["stable_diffusion"]
    # whisper_variants = whisper_variants + [f"{name}.en" for name in whisper_variants[:4]]
    parser = DefaultArgParser(["pytorch"])
    parser.require_model_name(stable_diffusion_variants)
    run_pytorch_fp32(**vars(parser.parse()))
