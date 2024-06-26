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


def run_pytorch(batch_size, num_runs, timeout, dataset_path, disable_jit_freeze=False, **kwargs):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.generation import GenerationConfig
    from utils.nlp.alpaca_instruct import AlpacaInstruct
    from utils.pytorch import PyTorchRunnerV2, apply_compile
    from utils.benchmark import run_model

    def run_single_pass(pytorch_runner, dataset):
        input_tensor = tokenizer.encode(dataset.get_input_string(), return_tensors="pt")
        input_tensor = torch.cat([input_tensor for _ in range(batch_size)], 0)
        config = GenerationConfig()
        config.max_new_tokens = 100
        config.do_sample = True
        output = pytorch_runner.run(inputs=input_tensor, generation_config=config)
        pytorch_runner.set_task_size(sum([len(output[i]) - len(input_tensor[i]) for i in range(batch_size)]))
        for i in range(batch_size):
            dataset.submit_prediction(tokenizer.decode(output[i][len(input_tensor[i]):], skip_special_tokens=True))

    model = AutoModelForCausalLM.from_pretrained("core42/jais-13b-chat", trust_remote_code=True)
    model.eval()
    model.generate = apply_compile(model.generate)

    tokenizer = AutoTokenizer.from_pretrained("core42/jais-13b-chat")
    dataset = AlpacaInstruct(batch_size, dataset_path=dataset_path)
    decode = lambda t: tokenizer.batch_decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=True)

    runner = PyTorchRunnerV2(model.generate)

    return run_model(run_single_pass, runner, dataset, 1, num_runs, timeout)


def run_pytorch_fp32(batch_size, num_runs, timeout, dataset_path, disable_jit_freeze=False, **kwargs):
    return run_pytorch(batch_size, num_runs, timeout, dataset_path, disable_jit_freeze, **kwargs)


def main():
    from utils.helpers import DefaultArgParser
    parser = DefaultArgParser(["pytorch"])
    parser.ask_for_batch_size()
    parser.add_argument("--dataset_path",
                        type=str,
                        help="path to JSON file with instructions")
    run_pytorch_fp32(**vars(parser.parse()))


if __name__ == "__main__":
    main()
