import os
import sys
import time

import torch
from torch import autocast
from tqdm import tqdm, trange
from PIL import Image
from omegaconf import OmegaConf
from einops import rearrange
import numpy as np
from torchvision.utils import make_grid

from utils.benchmark import run_model
from utils.pytorch import PyTorchRunnerV2
# from utils.text_to_image.stable_diffusion import StableDiffusion
from text_to_image.stable_diffusion.stablediffusion.scripts.txt2img import load_model_from_config
from text_to_image.stable_diffusion.stablediffusion.ldm.models.diffusion.ddim import DDIMSampler


def run_pytorch_fp32(model_name, num_runs, timeout):
    batch_size = 1
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "stablediffusion"))

    # checkpoint_link = "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/v2-1_512-ema-pruned.ckpt"
    checkpoint = "v2-1_512-ema-pruned.ckpt"
    config = OmegaConf.load("stablediffusion/configs/stable-diffusion/intel/v2-inference-fp32.yaml")
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

    outpath = "outputs/txt2img-samples"
    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    sample_count = 0
    base_count = len(os.listdir(sample_path))
    grid_count = len(os.listdir(outpath)) - 1

    data = [batch_size * [prompt]]
    sample_time = 0

    for i in [1, 2, 3, 4, 5, 6, 7, 8, 9]:
        # with torch.no_grad(), precision_scope(device), model.ema_scope():
        with torch.no_grad(), model.ema_scope():
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
