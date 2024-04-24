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


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run karpathy/nanoGPT with small modifications.")
    parser.add_argument("-m", "--model_name",
                        type=str, choices=["gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"], default="gpt2",
                        help="name of the model")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str, default="pytorch",
                        choices=["pytorch"],
                        help="specify the framework in which a model should be run")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--lambada_path",
                        type=str,
                        help="path to directory with the LAMBADA dataset")
    parser.add_argument("--disable_jit_freeze", action='store_true',
                        help="if true model will be run not in jit freeze mode")
    return parser.parse_args()


def run_pytorch(model_name, batch_size, num_runs, timeout, lambada_path, disable_jit_freeze=True, **kwargs):
    import tiktoken
    import torch
    from utils.pytorch import PyTorchRunner
    from utils.benchmark import run_model
    from utils.nlp.lambada import Lambada
    from nanoGPT.model import GPT

    def run_single_pass(pytorch_runner, lambada):
        start_ids = lambada.get_input_array()[0]
        input = (torch.tensor(start_ids, dtype=torch.long)[None, ...])
        output = pytorch_runner.run(batch_size, {"idx": input, "max_new_tokens": 5, "temperature": 1.0})
        output = decode(output[0].tolist())

        for i in range(batch_size):
            first_new_word = output.replace(decode(start_ids), '').split()[0]
            lambada.submit_prediction(i, first_new_word)

    torch.manual_seed(1337)
    model = GPT.from_pretrained(model_name, dict(dropout=0.0))
    enc = tiktoken.get_encoding("gpt2")
    encode = lambda s: enc.encode(s, allowed_special={"<|endoftext|>"})
    decode = lambda ll: enc.decode(ll)
    dataset = Lambada(batch_size, encode, decode, lambada_path)
    runner = PyTorchRunner(model, disable_jit_freeze=disable_jit_freeze, func="generate")

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch_fp32(model_name, batch_size, num_runs, timeout, lambada_path, disable_jit_freeze=True, **kwargs):
    run_pytorch(model_name, batch_size, num_runs, timeout, lambada_path, disable_jit_freeze, **kwargs)


def main():
    from utils.misc import print_goodbye_message_and_die
    args = parse_args()
    if args.framework == "pytorch":
        if args.batch_size != 1:
            raise ValueError("Batch size must be 1 for this model.")
        run_pytorch_fp32(**vars(args))
    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
