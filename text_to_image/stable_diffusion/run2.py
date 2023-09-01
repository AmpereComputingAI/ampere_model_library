import os
import sys

import cv2
import torch
import pathlib
import numpy as np
from PIL import Image
from pathlib import Path
from einops import rearrange
from torchvision.utils import make_grid
from utils.downloads.utils import get_downloads_path


# def put_watermark(img, wm_encoder=None):
#     if wm_encoder is not None:
#         img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
#         img = wm_encoder.encode(img, 'dwtDct')
#         img = Image.fromarray(img[:, :, ::-1])
#     return img


def run_pytorch_fp32(model_path, config, steps, scale, prompt, outdir, batch_size, num_runs, timeout):
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "stablediffusion"))

    from omegaconf import OmegaConf
    from contextlib import nullcontext
    from utils.benchmark import run_model
    from imwatermark import WatermarkEncoder
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

    # =========================
    # TODO: stuff for saving images, used to evaluate output, to be removed when accuracy measures are implemented
    # os.makedirs(outdir, exist_ok=True)
    # outpath = outdir
    # sample_path = os.path.join(outpath, "samples")
    # os.makedirs(sample_path, exist_ok=True)
    # print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    # wm = "SDV2"
    # wm_encoder = WatermarkEncoder()
    # wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    # =========================
    # TODO: torchscript stuff, should it stay here?
    unet = model.model.diffusion_model
    decoder = model.first_stage_model.decoder

    stablediffusion_data = pathlib.Path(get_downloads_path(), "stable_diffusion")
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
        _runner.run(batch_size * steps)
        _stablediffusion.submit_count()

    def wrapper():
        # all_samples = list()
        # base_count = len(os.listdir(sample_path))
        # sample_count = 0
        # grid_count = len(os.listdir(outpath)) - 1

        with torch.no_grad(), nullcontext(torch.device("cpu")), model.ema_scope():
            uc = model.get_learned_conditioning(batch_size * [""]) if scale != 1.0 else None
            #TODO: conditioning below needs 'prompt' to be a list of prompt * batch size, currently the batch size is
            # 1 and is fixed
            samples, _ = sampler.sample(S=steps,
                                        conditioning=model.get_learned_conditioning(prompt),
                                        batch_size=batch_size,
                                        shape=shape,
                                        verbose=False,
                                        unconditional_guidance_scale=scale,
                                        unconditional_conditioning=uc,
                                        eta=0.0,
                                        x_T=None)

        # x_samples = model.decode_first_stage(samples)
        # x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
        #
        # for x_sample in x_samples:
        #     x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
        #     img = Image.fromarray(x_sample.astype(np.uint8))
        #     img = put_watermark(img, wm_encoder)
        #     img.save(os.path.join(sample_path, f"{base_count:05}.png"))
        #     base_count += 1
        #     sample_count += 1
        #
        # all_samples.append(x_samples)
        #
        # # additionally, save as grid
        #
        # grid = torch.stack(all_samples, 0)
        # grid = rearrange(grid, 'n b c h w -> (n b) c h w')
        # grid = make_grid(grid, nrow=1)
        #
        # # to image
        # grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
        # grid = Image.fromarray(grid.astype(np.uint8))
        # grid = put_watermark(grid, wm_encoder)
        # grid.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
        # grid_count += 1
        #
        # print(f"Your samples are ready and waiting for you here: \n{outpath}")

    runner = PyTorchRunnerV2(wrapper)
    stablediffusion = StableDiffusion()

    return run_model(single_pass_pytorch, runner, stablediffusion, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser
    parser = DefaultArgParser(["pytorch"])
    parser.ask_for_batch_size()
    parser.require_model_path()
    parser.add_argument("--config", type=str,
                        default="stablediffusion/configs/stable-diffusion/intel/v2-inference-fp32.yaml",
                        help="path to config which constructs model")
    parser.add_argument("--steps", type=int, default=25, help="steps through which the model processes the input")
    parser.add_argument('--scale', type=int, default=9, help="scale of the image")
    parser.add_argument("--prompt", type=str, nargs="?",
                        default="a professional photograph of an astronaut riding a triceratops",
                        help="the prompt to render")
    parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to",
                        default="outputs/txt2img-samples")
    run_pytorch_fp32(**vars(parser.parse()))
