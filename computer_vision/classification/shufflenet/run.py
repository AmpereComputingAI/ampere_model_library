import argparse
import torch
import torchvision

from utils.cv.imagenet import ImageNet
from utils.pytorch import PyTorchRunner
from utils.ort import OrtRunner
from utils.benchmark import run_model
from utils.misc import UnsupportedPrecisionValueError, FrameworkUnsupportedError, ModelPathUnspecified


def parse_args():
    parser = argparse.ArgumentParser(description="Run ShuffleNet model.")
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
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
    parser.add_argument("--framework",
                        type=str,
                        choices=["pytorch", "ort"], required=True,
                        help="specify the framework in which a model should be run")
    parser.add_argument("--jit_freeze", action='store_true',
                        help="specify if model should be run with torch.jit.freeze model")
    return parser.parse_args()


def run_pytorch_fp(batch_size, num_of_runs, timeout, images_path, labels_path, jit_freeze):

    def run_single_pass(pytorch_runner, imagenet):
        shape = (224, 224)
        output = pytorch_runner.run(torch.from_numpy(imagenet.get_input_array(shape)))

        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output[i]),
                imagenet.extract_top5(output[i])
            )

    dataset = ImageNet(batch_size, "RGB", images_path, labels_path,
                       pre_processing='PyTorch', is1001classes=False, order='NCHW')
    runner = PyTorchRunner(torchvision.models.__dict__["shufflenet_v2_x1_0"](pretrained=True),
                           jit_freeze=jit_freeze)

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def run_pytorch_fp32(batch_size, num_of_runs, timeout, images_path, labels_path, jit_freeze):
    return run_pytorch_fp(batch_size, num_of_runs, timeout, images_path, labels_path, jit_freeze)

def run_ort_fp32(model_path, batch_size, num_of_runs, timeout, images_path, labels_path):

    def run_single_pass(ort_runner, imagenet):
        shape = (224, 224)
        ort_runner.set_input_tensor("gpu_0/data_0", imagenet.get_input_array(shape))
        output = ort_runner.run()
        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output[i]),
                imagenet.extract_top5(output[i])
            )

    dataset = ImageNet(batch_size, "RGB", images_path, labels_path,
                       pre_processing="Inception", is1001classes=False, order="NCHW")
    runner = OrtRunner(model_path)

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def main():
    args = parse_args()
    if args.framework == "pytorch":
        if args.precision == "fp32":
            run_pytorch_fp32(
                args.batch_size, args.num_runs, args.timeout, args.images_path, args.labels_path, args.jit_freeze
            )
        else:
            raise UnsupportedPrecisionValueError(args.precision)
    elif args.framework == "ort":
        if args.model_path is None:
            raise ModelPathUnspecified(args.model_path)
        if args.precision == "fp32":
            run_ort_fp32(
                args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.labels_path
            )
        else:
            raise UnsupportedPrecisionValueError(args.precision)
    else:
        raise FrameworkUnsupportedError(args.framework)


if __name__ == "__main__":
    main()