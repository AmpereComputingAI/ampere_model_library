# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
try:
    from utils import misc  # noqa
except ModuleNotFoundError:
    import os
    import sys
    filename = "set_env_variables.sh"
    directory = os.path.realpath(__file__).split("/")[:-1]
    for idx in range(1, len(directory) - 1):
        subdir = "/".join(directory[:-idx])
        if filename in os.listdir(subdir):
            print(f"\nPlease run \033[91m'source {os.path.join(subdir, filename)}'\033[0m first.")
            break
    else:
        print(f"\n\033[91mFAIL: Couldn't find {filename}, are you running this script as part of Ampere Model Library?"
              f"\033[0m")
    sys.exit(1)


def run_pytorch_fp32(model_name, steps, batch_size, num_runs, timeout, **kwargs):
    import torch._dynamo
    from diffusers import DiffusionPipeline
    torch._dynamo.config.suppress_errors = True

    from utils.benchmark import run_model
    from utils.pytorch import apply_compile
    from utils.pytorch import PyTorchRunnerV2
    from utils.text_to_image.stable_diffusion import StableDiffusion

    model = DiffusionPipeline.from_pretrained(model_name,
                                              use_safetensors=True).to("cpu")

    model.unet = apply_compile(model.unet)
    model.vae.decoder = apply_compile(model.vae.decoder)

    def single_pass_pytorch(_runner, _stablediffusion):
        prompts = [_stablediffusion.get_input() for _ in range(batch_size)]
        x_samples = _runner.run(batch_size * steps, prompt=prompts, num_inference_steps=steps)
        _stablediffusion.submit_count(batch_size, x_samples)

    runner = PyTorchRunnerV2(model)
    stable_diffusion_dataset = StableDiffusion()
    return run_model(single_pass_pytorch, runner, stable_diffusion_dataset, batch_size, num_runs, timeout)


def run_pytorch_bf16(model_name, steps, batch_size, num_runs, timeout, **kwargs):
    import torch._dynamo
    from diffusers import DiffusionPipeline
    torch._dynamo.config.suppress_errors = True

    from utils.benchmark import run_model
    from utils.pytorch import apply_compile
    from utils.pytorch import PyTorchRunnerV2
    from utils.text_to_image.stable_diffusion import StableDiffusion

    model = DiffusionPipeline.from_pretrained(model_name,
                                              use_safetensors=True,
                                              torch_dtype=torch.bfloat16).to("cpu")

    model.unet = apply_compile(model.unet)
    model.vae.decoder = apply_compile(model.vae.decoder)

    def single_pass_pytorch(_runner, _stablediffusion):
        prompts = [_stablediffusion.get_input() for _ in range(batch_size)]
        x_samples = _runner.run(batch_size * steps, prompt=prompts, num_inference_steps=steps)
        _stablediffusion.submit_count(batch_size, x_samples)

    runner = PyTorchRunnerV2(model)
    stable_diffusion_dataset = StableDiffusion()
    return run_model(single_pass_pytorch, runner, stable_diffusion_dataset, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser
    from utils.misc import print_goodbye_message_and_die

    stablediffusion_variants = ["stabilityai/stable-diffusion-xl-base-1.0"]
    parser = DefaultArgParser(["pytorch"])
    parser.require_model_name(stablediffusion_variants)
    parser.ask_for_batch_size()
    parser.add_argument("--steps", type=int, default=25, help="steps through which the model processes the input")
    parser.add_argument("-p", "--precision", type=str, choices=["fp32", "bf16"], required=True,
                        help="precision in which to run the model")

    args = parser.parse()
    if args.precision == "fp32":
        run_pytorch_fp32(**vars(parser.parse()))
    elif args.precision == "bf16":
        run_pytorch_bf16(**vars(parser.parse()))
    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified precision: " + args.precision)
