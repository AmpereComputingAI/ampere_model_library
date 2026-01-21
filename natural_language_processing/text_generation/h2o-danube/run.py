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

    pipe = pipeline("text-generation", model=model_name,
                    torch_dtype=torch.bfloat16, device_map="auto")

    pipe.model = apply_compile(pipe.model)

    def single_pass_pytorch(_runner, _dataset):
        prompt = encode([{"role": "user", "content": _dataset.get_input_string()}])
        response = _runner.run(1, prompt, max_new_tokens=256)
        _dataset.submit_prediction(response[0]["generated_text"])
        # print(res[0]["generated_text"])

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
