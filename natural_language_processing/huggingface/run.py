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
    parser = argparse.ArgumentParser(description="Run NLP model from Hugging Face Transformers repo.")
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


def get_benchmark_args(model, batch_size, sequence_length, fp16, ):
    return TensorFlowBenchmarkArguments(models=[model], batch_sizes=[batch_size], sequence_lengths=[sequence_length],
                                        fp16=False, repeat=1, num_runs=num_of_runs, timeout=timeout,
                                        inference=True, memory=False, eager_mode=False)


def run_tf_fp32(model, batch_size, sequence_length):
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


    benchmark = TensorFlowBenchmark(args)

    if profiler or os.environ.get('DLS_PROFILER', "0") == "1":
        tf.config.threading.set_inter_op_parallelism_threads(1)
        # disable_eager_execution()

    results = benchmark.run()


def run_tf_fp16(model, batch_size, sequence_length):
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

    benchmark = TensorFlowBenchmark(args)

    if profiler or os.environ.get('DLS_PROFILER', "0") == "1":
        tf.config.threading.set_inter_op_parallelism_threads(1)
        # disable_eager_execution()

    results = benchmark.run()


def main():
    args = parse_args()
    if args.precision == "fp32":
        run_tf_fp32(
            args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.labels_path
        )
    elif args.precision == "fp16":
        run_tf_fp16(
            args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.labels_path
        )
    else:
        assert False


if __name__ == "__main__":
    main()
