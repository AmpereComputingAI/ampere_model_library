import argparse

from utils.cv.kits import KiTS19
from tensorflow.python.saved_model import tag_constants
from utils.tf import TFSavedModelRunner
from utils.benchmark import run_model
import numpy as np
import time
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3D Unet model.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=True,
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
                        type=str, required=True,
                        help="path to directory with KiTS19 dataset")
    return parser.parse_args()


def run_tf_fp32(model_path, num_of_runs, timeout, images_path, anno_path, groundtruth_path):

    def run_single_pass(tf_runner, kits):
        output = tf_runner.run(tf.constant(np.expand_dims(kits.get_input_array(), axis=0)))
        output = output["output_0"]
        kits.submit_predictions(output)

    dataset = KiTS19(dataset_dir_path=images_path)
    runner = TFSavedModelRunner()
    saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
    runner.model = saved_model_loaded.signatures['serving_default']

    return run_model(run_single_pass, runner, dataset, 1, num_of_runs, timeout)


def main():
    args = parse_args()
    run_tf_fp32(
        args.model_path, args.num_runs, args.timeout, args.dataset_path
    )


if __name__ == "__main__":
    main()
