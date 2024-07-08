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


def run_pytorch(num_runs, timeout, dataset_path, disable_jit_freeze=False, **kwargs):
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
    from utils.nlp.alpaca_instruct import AlpacaInstruct
    from utils.pytorch import PyTorchRunnerV2, apply_compile
    from utils.benchmark import run_model

    def run_single_pass(pytorch_runner, dataset):
        input_array = [{"role": "user", "content": dataset.get_input_string()}]
        inputs = encode(input_array)
        
        outputs = pytorch_runner.run(inputs=inputs, generation_config=config)
        pytorch_runner.set_task_size(outputs.shape[1] - inputs.shape[1])
        response = decode(outputs[:, inputs.shape[1]:])[0]
        dataset.submit_prediction(response)

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    model.eval()
    model.forward = apply_compile(model.forward)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    dataset = AlpacaInstruct(1, dataset_path=dataset_path)
    encode = lambda i: tokenizer.apply_chat_template(i, return_tensors="pt")
    decode = lambda t: tokenizer.batch_decode(t, skip_special_tokens=True)
    config = GenerationConfig.from_pretrained("mistralai/Mixtral-8x7B-Instruct-v0.1")
    config.max_new_tokens=100
    config.do_sample = True
    config.pad_token_id = config.eos_token_id

    runner = PyTorchRunnerV2(model.generate)

    return run_model(run_single_pass, runner, dataset, 1, num_runs, timeout)


def run_pytorch_fp32(num_runs, timeout, dataset_path, disable_jit_freeze=False, **kwargs):
    return run_pytorch(num_runs, timeout, dataset_path, disable_jit_freeze, **kwargs)


def main():
    from utils.helpers import DefaultArgParser
    parser = DefaultArgParser(["pytorch"])
    parser.add_argument("--dataset_path",
                        type=str,
                        help="path to JSON file with instructions")
    run_pytorch_fp32(**vars(parser.parse()))


if __name__ == "__main__":
    main()
