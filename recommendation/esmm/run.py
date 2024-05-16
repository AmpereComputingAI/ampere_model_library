# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
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


def run_tf(model_path, batch_size, num_runs, timeout, dataset_path):
    import tensorflow as tf
    from utils.tf import TFSavedModelRunner
    from utils.benchmark import run_model
    from utils.recommendation.census_income import CensusIncome

    def single_pass_tf(tf_runner, dataset):
        inputs = {k: tf.convert_to_tensor(v) for k, v in dataset.get_inputs().items()}
        dataset.submit_results(tf_runner.run(batch_size, **inputs))

    runner = TFSavedModelRunner()
    ds = CensusIncome(batch_size, dataset_path)
    runner.model = tf.saved_model.load(model_path).signatures["serving_default"]
    return run_model(single_pass_tf, runner, ds, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser
    parser = DefaultArgParser(["tf"])
    parser.ask_for_batch_size(default_batch_size=2048)
    parser.require_model_path()
    parser.add_argument("--dataset_path",
                        type=str, required=True, help="path to csv file with 'Adult Census Income' data")
    run_tf(**vars(parser.parse()))
