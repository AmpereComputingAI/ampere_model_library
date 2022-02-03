import argparse

import numpy as np

from utils.cv.brats import BraTS19
from utils.tf import TFFrozenModelRunner
from utils.benchmark import run_model

from utils.misc import UnsupportedPrecisionValueError, FrameworkUnsupportedError


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3D Unet BraTS 2019 model.")
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--dataset_path",
                        type=str,
                        help="path to directory with BraTS19 dataset")
    parser.add_argument("--framework",
                        type=str,
                        choices=["tf"], required=True,
                        help="specify the framework in which a model should be run")
    return parser.parse_args()


def run_tf_fp(model_path, num_runs, timeout, dataset_path, **kwargs):

    def run_single_pass(tf_runner, brats):
        tf_runner.set_input_tensor("input:0", np.expand_dims(brats.get_input_array(), axis=0))
        output = tf_runner.run()
        brats.submit_predictions(
            output["output:0"]
        )

    dataset = BraTS19(dataset_dir_path=dataset_path)
    runner = TFFrozenModelRunner(model_path, ["output:0"])
    return run_model(run_single_pass, runner, dataset, 1, num_runs, timeout)


def run_tf_fp32(**kwargs):
    return run_tf_fp(**kwargs)


def run_tf_fp16(**kwargs):
    return run_tf_fp(**kwargs)


def main():
    args = parse_args()
    if args.framework == "tf":
        if args.model_path is None:
            print_goodbye_message_and_die(
                "a path to model is unspecified!")

        if args.precision == "fp32":
            run_tf_fp32(**vars(args))
        elif args.precision == "fp16":
            run_tf_fp16(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
