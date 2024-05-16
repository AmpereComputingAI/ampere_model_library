# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC

import torch
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

from utils.cv.kits import KiTS19
from utils.benchmark import run_model


try:
    from utils import misc  # noqa
except ModuleNotFoundError:
    import os
    import sys
    filename = "set_env_variables.sh"
    directory = os.path.realpath(__file__).split("/")[:-1]
    for idx in range(1, len(directory) - 1):
        subdir = "/".join(directory[:-idx])
        if filename in os.listdir(subdir):
            print(f"\nPlease run \033[91m'source {os.path.join(subdir, filename)}'\033[0m first.")
            break
    else:
        print(f"\n\033[91mFAIL: Couldn't find {filename}, are you running this script as part of Ampere Model Library?"
              f"\033[0m")
    sys.exit(1)


def run_tf_fp(model_path, num_runs, timeout, kits_path):
    import numpy as np
    import tensorflow as tf
    from tensorflow.python.saved_model import tag_constants
    from utils.cv.kits import KiTS19
    from utils.benchmark import run_model
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
    from utils.pytorch import PyTorchRunnerV2

    def run_single_pass(pytorch_runner, kits):
        output = pytorch_runner.run(1, torch.from_numpy(np.expand_dims(kits.get_input_array(), axis=0)))
        kits.submit_predictions(tf.convert_to_tensor(output.numpy()))

    dataset = KiTS19(dataset_dir_path=kits_path)
    model = torch.jit.load(model_path, map_location=torch.device('cpu')).eval()
    model = torch.jit.freeze(model)
    runner = PyTorchRunnerV2(model)

    return run_model(run_single_pass, runner, dataset, 1, num_runs, timeout)


def run_tf_fp32(model_path, num_runs, timeout, kits_path, **kwargs):
    return run_tf_fp(model_path, num_runs, timeout, kits_path)


def run_pytorch_fp32(model_path, num_runs, timeout, kits_path, **kwargs):
    return run_pytorch_fp(model_path, num_runs, timeout, kits_path)


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
