from utils.pytorch import apply_compile

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


def run_pytorch_fp32(model, steps, batch_size, num_runs, timeout, **kwargs):
    from utils.benchmark import run_model
    from diffusers import DiffusionPipeline
    from utils.pytorch import PyTorchRunnerV2
    from utils.text_to_image.stable_diffusion import StableDiffusion

    model = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",
                                              use_safetensors=True).to("cpu")
    model.unet = apply_compile(model.unet)

    def single_pass_pytorch(_runner, _stablediffusion):
        prompt = _stablediffusion.get_input()
        x_samples = _runner.run(batch_size * steps, prompt=prompt)

    runner = PyTorchRunnerV2(model)
    stablediffusion = StableDiffusion()
    return run_model(single_pass_pytorch, runner, stablediffusion, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser
    parser = DefaultArgParser(["pytorch"])