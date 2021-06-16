from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments, BertConfig
import tensorflow as tf
from tensorflow.python.profiler import model_analyzer
from datetime import datetime
import os
import argparse
import utils.misc as utils
from profiler1 import profiler1
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
    tf.DLS.force_enable_profiler()

    env_var = "PROFILER_LOG_DIR"
    logs_dir = utils.get_env_variable(
        env_var, f"Path to profiler log directory has not been specified with {env_var} flag")

    try:
        shutil.rmtree(os.environ['PROFILER_LOG_DIR'] + '/plugins/profile')
    except:
        print('some error occured')

    args = TensorFlowBenchmarkArguments(models=["bert-base-uncased"], batch_sizes=[batch_size],
                                        sequence_lengths=[sequence_length])

    benchmark = TensorFlowBenchmark(args)
    results = benchmark.run()

    print(results)
    profiler1(logs_dir)

    dls_profiler = os.environ['DLS_PROFILER']

    if dls_profiler == '1':
        tf.DLS.print_profile_data()


def main():
    args = parse_args()
    benchmark_bert_base_uncased(args.batch_size, args.sequence_length, args.profiler)


if __name__ == "__main__":
    main()
