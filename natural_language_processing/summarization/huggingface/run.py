import argparse
import os
import logging

import numpy as np
import torch
import torch._dynamo.config

from utils.pytorch import PyTorchRunner, PyTorchRunnerV2, apply_compile_maybe
from utils.benchmark import run_model
from transformers import AutoTokenizer, BartForConditionalGeneration
from utils.nlp.cnn_dailymail import CNN_DailyMail
from utils.misc import print_goodbye_message_and_die


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run model from Huggingface's transformers repo for summarization task.")
    parser.add_argument("-m", "--model_name",
                        type=str, default="sshleifer/distilbart-cnn-6-6",
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
    parser.add_argument("--cnn_dm_path",
                        type=str,
                        help="path to directory with the CNN/DailyMail dataset")
    parser.add_argument("--disable_jit_freeze", action='store_true',
                        help="if true model will be run not in jit freeze mode")
    return parser.parse_args()


def run_pytorch(model_name, batch_size, num_runs, timeout, cnn_dm_path, disable_jit_freeze=True, **kwargs):
    def run_single_pass(pytorch_runner, cnn_dm):
        input = torch.tensor(np.array(cnn_dm.get_input_ids_array(), dtype=np.int32))
        output = pytorch_runner.run(batch_size, input, max_new_tokens=60)

        for i in range(batch_size):
            cnn_dm.submit_prediction(
                i,
                detokenize(output[i])
            )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(text):
        return tokenizer(text, add_special_tokens=True)

    def detokenize(summary):
        return tokenizer.decode(summary)

    torch._dynamo.config.cache_size_limit = 128
    torch._dynamo.config.verbose = True # or env_var TORCHDYNAMO_VERBOSE=1
    torch._logging.set_logs(graph_breaks=True)

    model = BartForConditionalGeneration.from_pretrained(model_name, torchscript=True)
    aio_available = '_aio_profiler_print' in dir(torch._C) and os.environ.get("AIO_PROCESS_MODE") != "0"
    model.forward = apply_compile_maybe(model.forward, aio_available)
    dataset = CNN_DailyMail(batch_size, tokenize, detokenize, dataset_path=cnn_dm_path)
    runner = PyTorchRunnerV2(model.generate)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def main():
    args = parse_args()
    if args.framework == "pytorch":
        run_pytorch(**vars(args))
    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
