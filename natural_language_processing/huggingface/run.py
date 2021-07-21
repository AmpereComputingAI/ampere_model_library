import transformers

import tensorflow as tf
from tensorflow.python.profiler import model_analyzer
from datetime import datetime
import os
import argparse
from utils.misc import print_goodbye_message_and_die
from utils.profiling import set_profile_path, get_profile_path, summarize_tf_profiling
import shutil
import csv


def parse_args():
    parser = argparse.ArgumentParser(description="Run NLP model from Hugging Face Transformers repo.")
    parser.add_argument("-m", "--model_name",
                        type=str, required=True,
                        help="name of the model")
    parser.add_argument("-p,", "--precision",
                        type=str, choices=["fp32", "fp16"], required=True,
                        help="precision of the model")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=8,
                        help="batch size to feed the model with")
    parser.add_argument("-s", "--sequence_length",
                        type=int, default=8,
                        help="sequence length to feed the model with")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--profiler",
                        action="store_true",
                        help="enables TF profiler tracing")
    return parser.parse_args()


def get_TensorFlowBenchmarkArguments(model_name, batch_size, sequence_length, num_of_runs, timeout, profiler, fp16):
    return transformers.TensorFlowBenchmarkArguments(models=[model_name],
                                                     batch_sizes=[batch_size],
                                                     sequence_lengths=[sequence_length],
                                                     num_runs=num_of_runs,
                                                     timeout=timeout,
                                                     fp16=fp16,
                                                     profiler=profiler,
                                                     repeat=1, memory=False)


def parse_perf_metrics(output, model_name):
    inference_output = output[0][model_name]
    assert len(inference_output["bs"]) == 1
    assert len(inference_output["ss"]) == 1
    bs = inference_output["bs"][0]
    ss = inference_output["ss"][0]
    latency_in_s = inference_output["result"][bs][ss]
    if type(latency_in_s) is str:
        print_goodbye_message_and_die("Model seems to be unsupported by Transformers in requested config")

    latency_in_ms = latency_in_s * 1000
    instances_per_second = bs / latency_in_s

    print("\n Latency: {:.0f} ms".format(latency_in_ms))
    print(" Throughput: {:.2f} ips".format(instances_per_second))
    return {"lat_ms": latency_in_ms, "throughput": instances_per_second}


def run_tf_fp32(args):
    selected_args = (args.model_name, args.batch_size, args.sequence_length, args.num_runs, args.timeout, args.profiler)
    runner = transformers.TensorFlowBenchmark(get_TensorFlowBenchmarkArguments(*selected_args, fp16=False))
    return parse_perf_metrics(runner.run(), args.model_name)


def run_tf_fp16(args):
    selected_args = (args.model_name, args.batch_size, args.sequence_length, args.num_runs, args.timeout, args.profiler)
    runner = transformers.TensorFlowBenchmark(get_TensorFlowBenchmarkArguments(*selected_args, fp16=True))
    return parse_perf_metrics(runner.run(), args.model_name)


def main():
    tf.config.threading.set_inter_op_parallelism_threads(1)

    try:
        transformers.onspecta()
    except AttributeError:
        print_goodbye_message_and_die("OnSpecta's fork of Transformers repo is not installed.")

    args = parse_args()

    if args.profiler:
        set_profile_path(args.model_name)

    if args.precision == "fp32":
        run_tf_fp32(args)
    elif args.precision == "fp16":
        run_tf_fp16(args)
    else:
        assert False

    if args.profiler:
        summarize_tf_profiling()


if __name__ == "__main__":
    main()
