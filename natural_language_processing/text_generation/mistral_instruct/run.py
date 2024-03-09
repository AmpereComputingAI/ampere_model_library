# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import os
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer
from utils.nlp.alpaca_instruct import AlpacaInstruct
from utils.pytorch import PyTorchRunnerV2, apply_compile
from utils.benchmark import run_model


def run_pytorch(num_runs, timeout, dataset_path, disable_jit_freeze=False, **kwargs):

    def run_single_pass(pytorch_runner, dataset):
        input_array = dataset.get_input_array()
        input_array = [{"role": "user", "content": f"{input_array['instruction']} {input_array['input']}".strip()}]
        inputs = encode(input_array)
        outputs = pytorch_runner.run(inputs=inputs, max_new_tokens=100, do_sample=True)
        pytorch_runner.set_task_size(outputs.shape[1] - inputs.shape[1])
        response = decode(outputs[:, inputs.shape[1]:])[0]
        dataset.submit_prediction(response)

    model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    model.eval()
    aio_available = '_aio_profiler_print' in dir(torch._C) and os.environ.get("AIO_PROCESS_MODE") != "0"
    model.generate = apply_compile(model.generate, aio_available)

    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-Instruct-v0.2")
    dataset = AlpacaInstruct(1, dataset_path=dataset_path)
    encode = lambda i: tokenizer.apply_chat_template(i, return_tensors="pt")
    decode = lambda t: tokenizer.batch_decode(t, skip_special_tokens=True)

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
