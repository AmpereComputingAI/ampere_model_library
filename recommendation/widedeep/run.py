# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import argparse

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.core.framework import graph_pb2

from utils.benchmark import run_model
from utils.misc import print_goodbye_message_and_die
from recommendation.widedeep.widedeep import WideDeep


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument('-b', '--batch_size', type=int,
                        help='batch size for inference',
                        default=1,
                        dest='batch_size')
    parser.add_argument("-f", "--framework",
                        type=str, default="tf",
                        choices=["tf"],
                        help="specify the framework in which a model should be run")
    parser.add_argument("--timeout",
                        type=float, default=10.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--dataset_path",
                        type=str,
                        help="path to a dataset")
    parser.add_argument("--tfrecords_path",
                        type=str,
                        help="path to a tfrecords file")
    args = parser.parse_args()
    if args.framework == "tf" and args.model_path is None:
        parser.error(f"You need to specify the model path when using {args.framework} framework.")
    return parser.parse_args()


def initialize_graph(path_to_model: str):
    """
    A function initializing TF graph from frozen .pb model.
    :param path_to_model: str
    :return: TensorFlow graph
    """

    graph = ops.Graph()
    graph_def = graph_pb2.GraphDef()

    with graph.as_default():
        with tf.compat.v1.gfile.GFile(path_to_model, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def)
    return graph


def run_tf_fp(graph, model_path, batch_size, num_runs, timeout, dataset_path, tfrecords_path):
    from utils.tf import TFFrozenModelRunner

    def run_single_pass(tf_runner, widedeep):
        tf_runner.set_input_tensor('import/new_numeric_placeholder:0', widedeep.get_input_array()[0])
        tf_runner.set_input_tensor('import/new_categorical_placeholder:0', widedeep.get_input_array()[1])
        output = tf_runner.run()
        widedeep.submit_predictions(output)

    runner = TFFrozenModelRunner(model_path, ["import/import/head/predictions/probabilities:0"], graph)
    dataset = WideDeep(batch_size=batch_size, config=runner.config, runner=runner.graph,
                       dataset_path=dataset_path, tfrecords_path=tfrecords_path)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_tf_fp32(graph, model_path, batch_size, num_runs, timeout, dataset_path, tfrecords_path, **kwargs):
    return run_tf_fp(graph, model_path, batch_size, num_runs, timeout, dataset_path, tfrecords_path)


def main():
    args = parse_args()
    download_widedeep_processed_data(args.batch_size)

    if args.framework == "tf":
        if args.model_path is None:
            print_goodbye_message_and_die(
                "a path to model is unspecified!")

        if args.precision == "fp32":
            run_tf_fp32(initialize_graph(args.model_path), **vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
