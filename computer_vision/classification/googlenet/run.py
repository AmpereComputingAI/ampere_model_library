# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import argparse
import warnings

import torch
import torchvision

from utils.cv.imagenet import ImageNet
from utils.pytorch import PyTorchRunner
from utils.benchmark import run_model
from utils.misc import print_goodbye_message_and_die
warnings.filterwarnings("ignore")


def parse_args():
    parser = argparse.ArgumentParser(description="Run GoogLeNet model.")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str, default="pytorch",
                        choices=["pytorch"],
                        help="specify the framework in which a model should be run")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--images_path",
                        type=str,
                        help="path to directory with ImageNet validation images")
    parser.add_argument("--labels_path",
                        type=str,
                        help="path to file with validation labels")
    parser.add_argument("--disable_jit_freeze", action='store_true',
                        help="if true model will be run not in jit freeze mode")
    return parser.parse_args()


def run_pytorch_fp(model_name, batch_size, num_runs, timeout, images_path, labels_path, disable_jit_freeze=False):

    def run_single_pass(pytorch_runner, imagenet):
        shape = (224, 224)
        output = pytorch_runner.run(torch.from_numpy(imagenet.get_input_array(shape)))
        if not disable_jit_freeze:
            output = output[0]

        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output[i]),
                imagenet.extract_top5(output[i])
            )

    dataset = ImageNet(batch_size, "RGB", images_path, labels_path,
                       pre_processing='PyTorch', is1001classes=False, order='NCHW')
    runner = PyTorchRunner(torchvision.models.__dict__[model_name](pretrained=True),
                           disable_jit_freeze=disable_jit_freeze)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch_fp32(model_name, batch_size, num_runs, timeout, images_path, labels_path, disable_jit_freeze, **kwargs):
    return run_pytorch_fp(model_name, batch_size, num_runs, timeout, images_path, labels_path, disable_jit_freeze)


def main():
    args = parse_args()
    if args.framework == "pytorch":
        if args.precision == "fp32":
            run_pytorch_fp32(model_name="googlenet", **vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)
    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
