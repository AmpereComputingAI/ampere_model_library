from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments, BertConfig

import tensorflow as tf
from tensorflow.python.profiler import model_analyzer
from datetime import datetime
import os
import argparse
import utils.misc as utils
from profiler import print_profiler_results
import shutil


def parse_args():
    parser = argparse.ArgumentParser(description="Run bert-base-uncased")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=8,
                        help="batch size to feed the model with")
    parser.add_argument("--sequence_length",
                        type=int, default=8,
                        help="sequence length to feed the model with")
    parser.add_argument("--profiler",
                        action="store_const", const=True,
                        help="sequence length to feed the model with")
    return parser.parse_args()


def benchmark_bert_base_uncased(batch_size, sequence_length, profiler):

    # if profiler or os.environ['DLS_PROFILER'] == '1':
    if profiler or 'DLS_PROFILER' in os.environ and os.environ['DLS_PROFILER'] == '1':
        os.environ["PROFILER"] = "1"

        # set the logs
        env_var = "PROFILER_LOG_DIR"
        logs_dir = utils.get_env_variable(
            env_var, f"Path to profiler log directory has not been specified with {env_var} flag")
        # remove old logs from logs directory
        try:
            shutil.rmtree(logs_dir + '/plugins/profile')
        except FileNotFoundError:
            print('no logs to clear, moving on ...')

    args = TensorFlowBenchmarkArguments(models=["bert-base-uncased"], batch_sizes=[batch_size],
                                        sequence_lengths=[sequence_length], inference=True, memory=False)

    benchmark = TensorFlowBenchmark(args)
    results = benchmark.run()
    print(results)

    if 'PROFILER' in os.environ:
        print_profiler_results(logs_dir)


def main():
    args = parse_args()
    benchmark_bert_base_uncased(args.batch_size, args.sequence_length, args.profiler)


if __name__ == "__main__":
    main()
