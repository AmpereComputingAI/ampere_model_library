import os
import sys

import torch
from omegaconf import OmegaConf
from text_to_image.stable_diffusion.stablediffusion.scripts.txt2img import load_model_from_config
from text_to_image.stable_diffusion.stablediffusion.ldm.models.diffusion.ddim import DDIMSampler


def run_pytorch_fp32(args):

    print(args)
    print(type(args))
    print(args.batch_size)
    print(type(args.batch_size))
    quit()

    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "stablediffusion"))
    from utils.benchmark import run_model
    from utils.pytorch import PyTorchRunnerV2
    from utils.text_to_image.stable_diffusion import StableDiffusion
    from text_to_image.stable_diffusion.stablediffusion.scripts.txt2img import main

    config = OmegaConf.load(f"{args.config}")
    device = torch.device("cuda") if args.device == "cuda" else torch.device("cpu")
    model = load_model_from_config(config, f"{args.ckpt}", device)
    sampler = DDIMSampler(model, device=device)

    def single_pass_pytorch(_runner, _stablediffusion):
        # array = _stablediffusion.get_input()
        _stablediffusion.submit_count(
            _runner.run(1)
        )

    def wrapper():
        c = model.get_learned_conditioning(["a professional photograph of an astronaut riding a triceratops"])
        uc = None
        if args.scale != 1.0:
            uc = model.get_learned_conditioning(args.batch_size * [""])
        shape = [args.C, args.H // args.f, args.W // args.f]

        start_code = None
        if args.fixed_code:
            start_code = torch.randn([args.n_samples, args.C, args.H // args.f, args.W // args.f], device=device)

        samples, _ = sampler.sample(S=args.steps,
                                    conditioning=c,
                                    batch_size=args.n_samples,
                                    shape=shape,
                                    verbose=False,
                                    unconditional_guidance_scale=args.scale,
                                    unconditional_conditioning=uc,
                                    eta=args.ddim_eta,
                                    x_T=start_code)

    def sampler_wrapper(args):
        return main(args)

    runner = PyTorchRunnerV2(wrapper)
    stablediffusion = StableDiffusion()

    return run_model(single_pass_pytorch, runner, stablediffusion, args.batch_size, args.num_runs, args.timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser
    parser = DefaultArgParser(["pytorch"])
    parser.ask_for_batch_size()
    parser.add_argument("--steps", type=int, default=25, help="steps through which the model processes the input")
    parser.add_argument("-w", '--width', type=int, default=512, help="width of the image")
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
    parser.add_argument("--f", type=int, default=8, help="downsampling factor, most often 8 or 16")
    parser.add_argument("--device", type=str, help="Device on which Stable Diffusion will be run",
                        choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--n_iter", type=int, default=1, help="sample this often")
    parser.add_argument("--ddim_eta", type=float, default=0.0, help="ddim eta (eta=0.0 corresponds to deterministic sampling")

    run_pytorch_fp32(parser.parse())
