# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import argparse

from utils.benchmark import run_model
from utils.recommendation.widedeep import WideDeep
from utils.misc import print_goodbye_message_and_die, download_widedeep_processed_data


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument('--batch_size', type=int,
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
    return parser.parse_args()


def run_tf_fp(model_path, batch_size, num_runs, timeout, dataset_path, tfrecords_path):
    from utils.tf import TFFrozenModelRunner

    def run_single_pass(tf_runner, widedeep):
        tf_runner.set_input_tensor(['import/new_numeric_placeholder:0', 'import/new_categorical_placeholder:0'],
                                   widedeep.get_input_array())
        output = tf_runner.run()
        widedeep.submit_predictions(output)

    runner = TFFrozenModelRunner(model_path, ["import/import/head/predictions/probabilities:0"], True)
    dataset = WideDeep(batch_size=batch_size, config=runner.config, runner=runner.graph, dataset_path=dataset_path, tfrecords_path=tfrecords_path)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_tf_fp32(model_path, batch_size, num_runs, timeout, dataset_path, tfrecords_path, **kwargs):
    return run_tf_fp(model_path, batch_size, num_runs, timeout, dataset_path, tfrecords_path)


def main():
    args = parse_args()
    download_widedeep_processed_data(args.batch_size)

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
