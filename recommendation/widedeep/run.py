# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import os
import math
import argparse

from utils.benchmark import run_model
from utils.recommendation.widedeep import WideDeep
from utils.misc import print_goodbye_message_and_die


def parse_args():
    parser = argparse.ArgumentParser(description="Run WideDeep model.")
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str, default="tf",
                        choices=["tf"],
                        help="specify the framework in which a model should be run")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--dataset_path",
                        type=str, required=True,
                        help="path to a dataset")

    return parser.parse_args()


placeholder_list = ['import/new_numeric_placeholder:0', 'import/new_categorical_placeholder:0']
# input_tensor = [graph.get_tensor_by_name(name) for name in placeholder_list]

output_name = ["import/import/head/predictions/probabilities:0"]
# output_tensor = graph.get_tensor_by_name("import/" + output_name + ":0" )


def run_tf_fp(model_path, batch_size, num_runs, timeout, dataset_path):
    from utils.tf import TFFrozenModelRunner

    def run_single_pass(tf_runner, widedeep):
        res_dataset = widedeep.input_fn(False, batch_size)

        tf_runner.run1(res_dataset, widedeep.no_of_batches)
        output = tf_runner.run()

    dataset = WideDeep(batch_size, dataset_path)
    res_dataset = dataset.input_fn(False, batch_size)
    runner = TFFrozenModelRunner(model_path, output_name, res_dataset)



    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_tf_fp32(model_path, batch_size, num_runs, timeout, dataset_path, **kwargs):
    return run_tf_fp(model_path, batch_size, num_runs, timeout, dataset_path)


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
