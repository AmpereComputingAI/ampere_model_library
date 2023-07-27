import os

import torch
import transformers

from utils.nlp.alpaca_instruct import AlpacaInstruct
from utils.pytorch import PyTorchRunnerV2, apply_compile_maybe
from utils.benchmark import run_model


def run_pytorch(model_path, num_runs, timeout, dataset_path, disable_jit_freeze=False, **kwargs):

    def run_single_pass(pytorch_runner, dataset):
        inputs = encode(dataset.get_input_array())
        outputs = pytorch_runner.run(1, inputs=inputs.input_ids, max_new_tokens=100)
        pytorch_runner.update_last_task_size(outputs.shape[1] - inputs.input_ids.shape[1])
        response = decode(outputs[:, inputs.input_ids.shape[1]:])
        dataset.submit_prediction(response)


    model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
    model.eval()
    aio = '_aio_profiler_print' in dir(torch._C)
    model = apply_compile_maybe(model, aio)

    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    dataset = AlpacaInstruct(1, dataset_path=dataset_path)
    encode = lambda i: tokenizer(i, return_tensors="pt")
    decode = lambda t: tokenizer.batch_decode(t, skip_special_tokens=True)[0]

    runner = PyTorchRunnerV2(model.generate)

    return run_model(run_single_pass, runner, dataset, 1, num_runs, timeout)


def run_pytorch_fp32(model_path, num_runs, timeout, dataset_path, disable_jit_freeze=False, **kwargs):
    run_pytorch(model_path, num_runs, timeout, dataset_path, disable_jit_freeze, **kwargs)


def main():
    from utils.helpers import DefaultArgParser
    parser = DefaultArgParser(["pytorch"])
    parser.require_model_path()
    parser.add_argument("--dataset_path",
                    type=str,
                    help="path to JSON file with instructions")
    run_pytorch_fp32(**vars(parser.parse()))


if __name__ == "__main__":
    main()
