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


def run_pytorch_fp32(model_name, num_runs, timeout, dataset_path, **kwargs):
    import torch
    from transformers import pipeline

    from utils.benchmark import run_model
    from utils.pytorch import apply_compile
    from utils.pytorch import PyTorchRunnerV2
    from utils.nlp.alpaca_instruct import AlpacaInstruct

    # model = DiffusionPipeline.from_pretrained(model_name,
    #                                          use_safetensors=True,
    #                                          torch_dtype=torch.bfloat16).to("cpu")

    pipe = pipeline("text-generation", model="h2oai/h2o-danube2-1.8b-chat",
                    torch_dtype=torch.bfloat16, device_map="auto")

    # messages = [{"role": "user", "content": "Why is drinking water so healthy?"}]
    # prompt = pipe.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    # res = pipe(prompt, max_new_tokens=256)

    # print(res[0]["generated_text"])

    # model.unet = apply_compile(model.unet)

    def single_pass_pytorch(_runner, _dataset):
        prompt = encode([{"role": "user", "content": _dataset.get_input_string()}])
        # prompts = [_stablediffusion.get_input() for _ in range(batch_size)]
        res = _runner.run(1, prompt=prompt, max_new_tokens=256)
        print(res[0]["generated_text"])

    runner = PyTorchRunnerV2(pipe)

    dataset = AlpacaInstruct(1, dataset_path=dataset_path)
    tokenizer = pipe.tokenizer.apply_chat_template
    encode = lambda i: tokenizer(i, tokenize=False, add_generation_prompt=True)
    return run_model(single_pass_pytorch, runner, dataset, 1, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser

    h2o_danube_variants = ["h2oai/h2o-danube2-1.8b-chat"]
    parser = DefaultArgParser(["pytorch"])
    parser.require_model_name(h2o_danube_variants)
    parser.add_argument("--dataset_path",
                        type=str,
                        help="path to JSON file with instructions")

    run_pytorch_fp32(**vars(parser.parse()))
