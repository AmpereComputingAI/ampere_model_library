from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments, BertConfig

import tensorflow as tf
from tensorflow.python.profiler import model_analyzer
from datetime import datetime
import os
import argparse
import utils.misc as utils
from profiler import print_profiler_results
import shutil
import csv


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark NLP models")
    parser.add_argument("-m", "--model",
                        type=str, default='bert-base-uncased',
                        help="batch size to feed the model with")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=8,
                        help="batch size to feed the model with")
    parser.add_argument("--sequence_length",
                        type=int, default=8,
                        help="sequence length to feed the model with")
    parser.add_argument("--profiler",
                        action="store_const", const=True,
                        help="Run the model with Tensorflow profiler")
    parser.add_argument("--precision",
                        type=str, default="fp32", choices=["fp32", "fp16"],
                        help="precision of the model")
    return parser.parse_args()


def benchmark_nlp_model(model, batch_size, sequence_length, profiler, precision):

    print(profiler)
    if precision == "fp32":
        fp16 = False
    else:
        fp16 = True

    if profiler or os.environ.get('DLS_PROFILER', "0") == "1":
        os.environ["PROFILER"] = "1"

        env_var = "PROFILER_LOG_DIR"
        logs_dir = utils.get_env_variable(
            env_var, f"Path to profiler log directory has not been specified with {env_var} flag")

        # set the logs
        # remove old logs from logs directory
        try:
            shutil.rmtree(logs_dir + '/plugins/profile')
        except FileNotFoundError:
            print('no logs to clear, moving on ...')

    # if profiler or os.environ['DLS_PROFILER'] == '1':
    args = TensorFlowBenchmarkArguments(models=[model], batch_sizes=[batch_size],
                                        sequence_lengths=[sequence_length], inference=True, memory=False,
                                        fp16=fp16)

    benchmark = TensorFlowBenchmark(args)
    results = benchmark.run()

    print(results)


def main():
    args = parse_args()
    benchmark_nlp_model(args.model, args.batch_size, args.sequence_length, args.profiler, args.precision)


if __name__ == "__main__":
    main()
