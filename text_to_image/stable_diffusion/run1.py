import os
import sys

import torch
from pathlib import Path


def run_pytorch_fp32(args):
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "stablediffusion"))

    from omegaconf import OmegaConf
    from contextlib import nullcontext

    from utils.benchmark import run_model
    from utils.pytorch import PyTorchRunnerV2
    from utils.text_to_image.stable_diffusion import StableDiffusion
    from text_to_image.stable_diffusion.stablediffusion.scripts.txt2img import load_model_from_config
    from text_to_image.stable_diffusion.stablediffusion.ldm.models.diffusion.ddim import DDIMSampler

    C = args.C
    H = args.H
    f = args.f
    W = args.W

    # n_samples = args.n_samples
    batch_size = args.batch_size
    steps = args.steps
    scale = args.scale
    ddim_eta = args.ddim_eta
    fixed_code = args.fixed_code
    prompt = args.prompt

    config = OmegaConf.load(f"{args.config}")
    device = torch.device("cuda") if args.device == "cuda" else torch.device("cpu")
    model = load_model_from_config(config, f"{args.ckpt}", device)
    sampler = DDIMSampler(model, device=device)

    unet = model.model.diffusion_model
    decoder = model.first_stage_model.decoder
    additional_context = nullcontext()
    shape = [C, H // f, W // f]

    start_code = None
    if fixed_code:
        start_code = torch.randn([batch_size, C, H // f, W // f], device=device)
    uc = None
    if scale != 1.0:
        uc = model.get_learned_conditioning(batch_size * [""])

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

    print("Running a forward pass to initialize optimizations")

    with torch.no_grad(), additional_context:
        for _ in range(3):
            c = model.get_learned_conditioning(prompt)
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

    def single_pass_pytorch(_runner, _stablediffusion):
        _runner.run(batch_size * steps)
        _stablediffusion.submit_count()

    def wrapper():
        samples, _ = sampler.sample(S=steps,
                                    conditioning=model.get_learned_conditioning([prompt]),
                                    batch_size=batch_size,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=scale,
                                    unconditional_conditioning=uc,
                                    eta=ddim_eta,
                                    x_T=start_code)

    runner = PyTorchRunnerV2(wrapper)
    stablediffusion = StableDiffusion()

    return run_model(single_pass_pytorch, runner, stablediffusion, args.batch_size, args.num_runs, args.timeout)


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

    run_pytorch_fp32(parser.parse())
