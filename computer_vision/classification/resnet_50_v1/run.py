import argparse

import numpy as np

from utils.benchmark import run_model
from utils.cv.imagenet import ImageNet
from utils.pytorch import PyTorchRunner

PYTORCH_MODEL_NAME = 'resnet50'


def parse_args():
    parser = argparse.ArgumentParser(description="Run ResNet-50 v1 model.")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
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
    return parser.parse_args()


def run_torch_fp32(batch_size, num_of_runs, timeout, images_path, labels_path):

    def run_single_pass(pytorch_runner, imagenet):
        shape = (224, 224)
        output = pytorch_runner.run(imagenet.get_input_array(shape))

        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output[i]),
                imagenet.extract_top5(output[i])
            )

    dataset = ImageNet(batch_size, "RGB", images_path, labels_path,
                       pre_processing='Inception', is1001classes=False, order='NCHW')
    runner = PyTorchRunner(PYTORCH_MODEL_NAME)

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def main():
    args = parse_args()
    if args.precision == "fp32":
        run_torch_fp32(
            args.batch_size, args.num_runs, args.timeout, args.images_path, args.labels_path
        )
    else:
        assert False, f"Behaviour undefined for precision {args.precision}"


if __name__ == "__main__":
    main()
