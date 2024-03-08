# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

# import argparse

import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from utils.cv.kits import KiTS19
from utils.benchmark import run_model


def run_tf_fp(model_path, num_runs, timeout, kits_path):
    from utils.tf import TFSavedModelRunner

    def run_single_pass(tf_runner, kits):
        output = tf_runner.run(1, tf.constant(np.expand_dims(kits.get_input_array(), axis=0)))
        output = output["output_0"]
        kits.submit_predictions(output)

    dataset = KiTS19(dataset_dir_path=kits_path)
    runner = TFSavedModelRunner()
    saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
    runner.model = saved_model_loaded.signatures['serving_default']

    return run_model(run_single_pass, runner, dataset, 1, num_runs, timeout)


def run_pytorch_fp(model_path, num_runs, timeout, kits_path):
    from utils.tf import TFSavedModelRunner

    def run_single_pass(tf_runner, kits):
        output = tf_runner.run(1, tf.constant(np.expand_dims(kits.get_input_array(), axis=0)))
        output = output["output_0"]
        kits.submit_predictions(output)

    dataset = KiTS19(dataset_dir_path=kits_path)
    runner = TFSavedModelRunner()
    saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
    runner.model = saved_model_loaded.signatures['serving_default']

    return run_model(run_single_pass, runner, dataset, 1, num_runs, timeout)


def run_tf_fp32(model_path, num_runs, timeout, kits_path, **kwargs):
    return run_tf_fp(model_path, num_runs, timeout, kits_path)


def run_pytorch_fp32(model_path, num_runs, timeout, kits_path, **kwargs):
    return run_tf_fp(model_path, num_runs, timeout, kits_path)


def main():
    from utils.helpers import DefaultArgParser
    parser = DefaultArgParser(["tf", "pytorch"])
    parser.require_model_path()
    parser.add_argument("--kits_path",
                        type=str,
                        help="path to directory with KiTS19 dataset")

    args = parser.parse()
    if args.framework == 'tf':
        run_tf_fp32(**vars(parser.parse()))
    else:
        run_pytorch_fp32(**vars(parser.parse()))


if __name__ == "__main__":
    main()
