# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import argparse

from utils.cv.openimages import OpenImagesDataset
from utils.benchmark import run_model
from utils.misc import print_goodbye_message_and_die


def parse_args():
    parser = argparse.ArgumentParser(description="Run Retinanet model.")
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
                        help="path to directory with OpenImages validation images")
    parser.add_argument("--anno_path",
                        type=str,
                        help="path to file with validation annotations")
    parser.add_argument("--disable_jit_freeze", action='store_true',
                        help="if true model will be run not in jit freeze mode")
    return parser.parse_args()


def run_pytorch_fp(model_path, batch_size, num_runs, timeout, images_path, anno_path, disable_jit_freeze=False):
    import torch
    import torchvision
    from utils.pytorch import PyTorchRunner

    def run_single_pass(pytorch_runner, openimages):
        shape = (800, 800)
        x = torch.tensor(openimages.get_input_array(shape))
        output = pytorch_runner.run(x)

        for i in range(batch_size):
            for d in range(output[i]['boxes'].shape[0]):
                openimages.submit_bbox_prediction(
                    i,
                    openimages.convert_bbox_to_coco_order(output[i]['boxes'][d].tolist()),
                    output[i]['scores'][d].item(),
                    output[i]['labels'][d].item()
                )

    dataset = OpenImagesDataset(batch_size, "RGB", images_path, anno_path,
                                pre_processing="YOLO", sort_ascending=True, order="NCHW")
    model = torch.jit.load(model_path)
    runner = PyTorchRunner(model, disable_jit_freeze=True)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch_fp32(model_path, batch_size, num_runs, timeout, images_path, anno_path, disable_jit_freeze, **kwargs):
    return run_pytorch_fp(model_path, batch_size, num_runs, timeout, images_path, anno_path, disable_jit_freeze)


def main():
    args = parse_args()

    if args.framework == "pytorch":
        if args.precision == "fp32":
            run_pytorch_fp32(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
