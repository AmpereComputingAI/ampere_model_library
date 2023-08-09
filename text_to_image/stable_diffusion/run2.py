import os
import sys

import cv2
import torch
from torch import autocast
from pathlib import Path
from itertools import islice
from tqdm import tqdm, trange
from einops import rearrange
from PIL import Image
import numpy as np
from torchvision.utils import make_grid


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def run_pytorch_fp32(args):
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "stablediffusion"))

    from omegaconf import OmegaConf
    from contextlib import nullcontext

    from utils.benchmark import run_model
    from pytorch_lightning import seed_everything
    from imwatermark import WatermarkEncoder
    from utils.pytorch import PyTorchRunnerV2
    from utils.text_to_image.stable_diffusion import StableDiffusion
    from text_to_image.stable_diffusion.stablediffusion.scripts.txt2img import load_model_from_config
    from text_to_image.stable_diffusion.stablediffusion.ldm.models.diffusion.ddim import DDIMSampler
    from text_to_image.stable_diffusion.stablediffusion.ldm.models.diffusion.plms import PLMSSampler
    from text_to_image.stable_diffusion.stablediffusion.ldm.models.diffusion.dpm_solver import DPMSolverSampler

    C = args.C
    H = args.H
    f = args.f
    W = args.W

    n_samples = args.n_samples
    n_rows = args.n_rows
    batch_size = args.batch_size
    steps = args.steps
    scale = args.scale
    ddim_eta = args.ddim_eta
    fixed_code = args.fixed_code
    prompt = args.prompt

    config = args.config
    device = args.device
    ckpt = args.ckpt
    seed = args.seed
    plms = args.plms
    dpm = args.dpm
    outdir = args.outdir
    from_file = args.from_file
    repeat = args.repeat
    torchscript = args.torchscript
    ipex = args.ipex
    bf16 = args.bf16
    precision = args.precision
    n_iter = args.n_iter

    seed_everything(seed)

    config = OmegaConf.load(f"{config}")
    device = torch.device("cuda") if device == "cuda" else torch.device("cpu")
    model = load_model_from_config(config, f"{ckpt}", device)

    if plms:
        sampler = PLMSSampler(model, device=device)
    elif dpm:
        sampler = DPMSolverSampler(model, device=device)
    else:
        sampler = DDIMSampler(model, device=device)

    os.makedirs(outdir, exist_ok=True)
    outpath = outdir

    print("Creating invisible watermark encoder (see https://github.com/ShieldMnt/invisible-watermark)...")
    wm = "SDV2"
    wm_encoder = WatermarkEncoder()
    wm_encoder.set_watermark('bytes', wm.encode('utf-8'))

    batch_size = n_samples
    n_rows = n_rows if n_rows > 0 else batch_size
    if not from_file:
        prompt = prompt
        assert prompt is not None
        data = [batch_size * [prompt]]

    else:
        print(f"reading prompts from {from_file}")
        with open(from_file, "r") as f:
            data = f.read().splitlines()
            data = [p for p in data for i in range(repeat)]
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    sample_count = 0
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    start_code = None
    if fixed_code:
        start_code = torch.randn([n_samples, C, H // f, W // f], device=device)

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
    with torch.no_grad(), \
            precision_scope(device), \
            model.ema_scope():
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
                samples, _ = sampler.sample(S=steps,
                                            conditioning=c,
                                            batch_size=n_samples,
                                            shape=shape,
                                            verbose=False,
                                            unconditional_guidance_scale=scale,
                                            unconditional_conditioning=uc,
                                            eta=ddim_eta,
                                            x_T=start_code)

                x_samples = model.decode_first_stage(samples)
                x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)

                for x_sample in x_samples:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img = put_watermark(img, wm_encoder)
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
        grid = put_watermark(grid, wm_encoder)
        grid.save(os.path.join(outpath, f'grid-{grid_count:04}.png'))
        grid_count += 1

    print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
          f" \nEnjoy.")


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser
    parser = DefaultArgParser(["pytorch"])
    parser.ask_for_batch_size()
    parser.add_argument("--steps", type=int, default=25, help="steps through which the model processes the input")
    parser.add_argument('--scale', type=int, default=9, help="scale of the image")
    parser.add_argument("--seed", type=int, default=42, help="the seed (for reproducible sampling)",)
    parser.add_argument("--config", type=str,
                        default="stablediffusion/configs/stable-diffusion/intel/v2-inference-fp32.yaml",
                        help="path to config which constructs model")
    parser.add_argument("--ckpt", type=str, required=True, help="path to checkpoint of model")
    parser.add_argument("--outdir", type=str, nargs="?", help="dir to write results to", default="outputs/txt2img-samples")
    parser.add_argument("--n_samples", type=int, default=1, help="how many samples to produce for each given prompt. A.k.a batch size",)
    parser.add_argument("--n_rows", type=int, default=1, help="rows in the grid (default: n_samples)")
    parser.add_argument("--prompt", type=str, nargs="?",
                        default="a professional photograph of an astronaut riding a triceratops",
                        help="the prompt to render")
    parser.add_argument("--H", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--W", type=int, default=512, help="image width, in pixel space")
    parser.add_argument("--C", type=int, default=4, help="latent channels")
    parser.add_argument("--f", type=int, default=8, help="down-sampling factor, most often 8 or 16")
    parser.add_argument("--device", type=str, help="Device on which Stable Diffusion will be run",
                        choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    parser.add_argument("--fixed_code", action='store_true', help="if enabled, uses the same starting code across all samples ")

    parser.add_argument("--ipex", action='store_true', help="Use Intel® Extension for PyTorch*")
    parser.add_argument("--precision", type=str, help="evaluate at this precision", choices=["full", "autocast"], default="autocast")
    parser.add_argument("--bf16", action='store_true', help="use bfloat16")
    parser.add_argument("--n_iter", type=int, default=3, help="sample this often")

    run_pytorch_fp32(parser.parse())
