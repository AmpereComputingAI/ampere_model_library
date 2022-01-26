import argparse
import os

import numpy as np
import pickle
import torch

import utils.cv.nnUNet.nnunet as nnunet
from utils.cv.brats import BraTS19
from utils.tf import TFFrozenModelRunner
from utils.pytorch import PyTorchRunner
from utils.benchmark import run_model
from utils.cv.nnUNet.nnunet.training.model_restore import recursive_find_python_class

from utils.misc import UnsupportedPrecisionValueError, FrameworkUnsupportedError


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3D Unet BraTS 2019 model.")
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
                        type=str,
                        help="path to directory with BraTS19 dataset")
    parser.add_argument("--framework",
                        type=str,
                        choices=["tf", "pytorch"], required=True,
                        help="specify the framework in which a model should be run")
    return parser.parse_args()


def run_tf_fp(model_path, num_of_runs, timeout, dataset_path):

    def run_single_pass(tf_runner, brats):
        tf_runner.set_input_tensor("input:0", np.expand_dims(brats.get_input_array(), axis=0))
        output = tf_runner.run()
        brats.submit_predictions(
            output["output:0"]
        )

    dataset = BraTS19(dataset_dir_path=dataset_path)
    runner = TFFrozenModelRunner(model_path, ["output:0"])
    return run_model(run_single_pass, runner, dataset, 1, num_of_runs, timeout)


def run_tf_fp32(model_path, num_of_runs, timeout, dataset_path):
    return run_tf_fp(model_path, num_of_runs, timeout, dataset_path)


def run_tf_fp16(model_path, num_of_runs, timeout, dataset_path):
    return run_tf_fp(model_path, num_of_runs, timeout, dataset_path)

def run_pytorch_fp32(model_path, num_of_runs, timeout, dataset_path):

    def run_single_pass(pytorch_runner, brats):
        output = pytorch_runner.run(torch.from_numpy(np.expand_dims(brats.get_input_array(), axis=0)))
        output = np.asarray(output[0])
        brats.submit_predictions(
            output
        )

    dataset = BraTS19(dataset_dir_path=dataset_path)
    model = restore_model(model_path)

    runner = PyTorchRunner(model.network)
    return run_model(run_single_pass, runner, dataset, 1, num_of_runs, timeout)

def restore_model(checkpoint):
    pkl_file = checkpoint + ".pkl"
    with open(pkl_file, 'rb') as f:
        info = pickle.load(f)
    init = info['init']
    name = info['name']
    search_in = os.path.join(nnunet.__path__[0], "training", "network_training")
    tr = recursive_find_python_class([search_in], name, current_module="nnunet.training.network_training")
    trainer = tr(*init)
    trainer.output_folder = f"{os.getcwd()}/result"
    trainer.output_folder_base = "/result"
    trainer.process_plans(info['plans'])
    if checkpoint is not None:
        trainer.load_checkpoint(checkpoint, False)
    return trainer


def main():
    args = parse_args()
    if args.framework == "tf":
        if args.precision == "fp32":
            run_tf_fp32(args.model_path, args.num_runs, args.timeout, args.dataset_path)
        elif args.precision == "fp16":
            run_tf_fp16(args.model_path, args.num_runs, args.timeout, args.dataset_path)
        else:
            raise UnsupportedPrecisionValueError(args.precision)
    elif args.framework == "pytorch":
        if args.precision == "fp32":
            run_pytorch_fp32(args.model_path, args.num_runs, args.timeout, args.dataset_path)
        else:
            raise UnsupportedPrecisionValueError(args.precision)
    else:
        raise FrameworkUnsupportedError(args.framework)


if __name__ == "__main__":
    main()
