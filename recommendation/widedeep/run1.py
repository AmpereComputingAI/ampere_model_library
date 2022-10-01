# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC


import time
import math
import argparse

import tensorflow as tf

from utils.tf import TFFrozenModelRunner
from utils.recommendation.widedeep import WideDeep
from utils.misc import print_goodbye_message_and_die


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument('--batch_size', type=int,
                        help='batch size for inference.Default is 512',
                        default=512,
                        dest='batch_size')
    parser.add_argument("-f", "--framework",
                        type=str, default="tf",
                        choices=["tf"],
                        help="specify the framework in which a model should be run")
    parser.add_argument('--num_intra_threads', type=int,
                        help='number of threads for an operator',
                        required=False,
                        default=28,
                        dest='num_intra_threads')
    parser.add_argument('--num_inter_threads', type=int,
                        help='number of threads across operators',
                        required=False,
                        default=2,
                        dest='num_inter_threads')
    parser.add_argument('--num_omp_threads', type=str,
                        help='number of threads to use',
                        required=False,
                        default=None,
                        dest='num_omp_threads')
    parser.add_argument("--dataset_path",
                        type=str, required=True,
                        help="path to a dataset")
    return parser.parse_args()


def run_tf_fp(model_path, batch_size, dataset_path):

    runner = TFFrozenModelRunner(model_path, ["import/import/head/predictions/probabilities:0"])
    dataset = WideDeep(batch_size, dataset_path, runner.config, runner.graph)

    placeholder_list = ['import/new_numeric_placeholder:0', 'import/new_categorical_placeholder:0']
    input_tensor = [runner.graph.get_tensor_by_name(name) for name in placeholder_list]
    output_tensor = runner.graph.get_tensor_by_name("import/import/head/predictions/probabilities:0")

    features_list = dataset.get_features_list(runner.config, runner.graph, dataset.no_of_batches)
    test = runner.test(runner.config, runner.graph, input_tensor, output_tensor, features_list)
    runner.set_input_tensor1(['import/new_numeric_placeholder:0', 'import/new_categorical_placeholder:0'],
                             dataset.get_input())
    # something = runner.run1(runner.config, runner.graph, dataset.no_of_batches)

    total_infer_consume = 0.0
    warm_iter = 100
    # Tensor("import/import/head/predictions/probabilities:0", shape=(None, 2), dtype=float32)
    output_tensor = runner.graph.get_tensor_by_name("import/import/head/predictions/probabilities:0")
    with tf.compat.v1.Session(config=runner.config, graph=runner.graph) as sess1:
        i = 0
        while True:
            if i >= dataset.no_of_batches:
                break
            if i > warm_iter:
                start = time.time()
                logistic = sess1.run(output_tensor, dict(zip(input_tensor, features_list[i][0:2])))
                finish = time.time()
            if i > warm_iter:
                infer_time = finish - start
                total_infer_consume += infer_time

            i = i + 1

    # evaluate_duration = total_infer_consume
    # latency = (1000 * batch_size * float(evaluate_duration) / float(dataset.no_of_test_samples - warm_iter * batch_size))
    # throughput = (dataset.no_of_test_samples - warm_iter * batch_size) / evaluate_duration
    #
    # print('--------------------------------------------------')
    # print('Total test records           : ', dataset.no_of_test_samples)
    # print('Batch size is                : ', batch_size)
    # print('Number of batches            : ', int(dataset.no_of_batches))
    # print('Inference duration (seconds) : ', round(evaluate_duration, 4))
    # print('Average Latency (ms/batch)   : ', round(latency, 4))
    # print('Throughput is (records/sec)  : ', round(throughput, 3))
    # print('--------------------------------------------------')


def run_tf_fp32(model_path, batch_size, dataset_path, **kwargs):
    return run_tf_fp(model_path, batch_size, dataset_path)


def main():
    args = parse_args()

    if args.framework == "tf":
        if args.model_path is None:
            print_goodbye_message_and_die(
                "a path to model is unspecified!")

        if args.precision == "fp32":
            run_tf_fp32(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
