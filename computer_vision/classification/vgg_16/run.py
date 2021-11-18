import argparse
import torch
import torchvision

from utils.cv.imagenet import ImageNet
from utils.tf import TFFrozenModelRunner
from utils.tflite import TFLiteRunner
from utils.pytorch import PyTorchRunner
from utils.benchmark import run_model
from utils.misc import UnsupportedPrecisionValueError, ModelPathUnspecified, FrameworkUnsupportedError


def parse_args():
    parser = argparse.ArgumentParser(description="Run VGG-16 model.")
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32", "fp16", "int8"], required=True,
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
                        choices=["pytorch", "tf"], required=True,
                        help="specify the framework in which a model should be run")
    return parser.parse_args()


def run_tf_fp(model_path, batch_size, num_of_runs, timeout, images_path, labels_path):

    def run_single_pass(tf_runner, imagenet):
        shape = (224, 224)
        tf_runner.set_input_tensor("input:0", imagenet.get_input_array(shape))
        output = tf_runner.run()
        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output["vgg_16/fc8/squeezed:0"][i]),
                imagenet.extract_top5(output["vgg_16/fc8/squeezed:0"][i])
            )

    dataset = ImageNet(batch_size, "RGB", images_path, labels_path,
                       pre_processing="VGG", is1001classes=False)
    runner = TFFrozenModelRunner(model_path, ["vgg_16/fc8/squeezed:0"])

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def run_pytorch_fp(batch_size, num_of_runs, timeout, images_path, labels_path):

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
    runner = PyTorchRunner(torchvision.models.__dict__["vgg16"](pretrained=True))

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def run_tf_fp32(model_path, batch_size, num_of_runs, timeout, images_path, labels_path):
    return run_tf_fp(model_path, batch_size, num_of_runs, timeout, images_path, labels_path)


def run_tf_fp16(model_path, batch_size, num_of_runs, timeout, images_path, labels_path):
    return run_tf_fp(model_path, batch_size, num_of_runs, timeout, images_path, labels_path)


def run_pytorch_fp32(batch_size, num_of_runs, timeout, images_path, labels_path):
    return run_pytorch_fp(batch_size, num_of_runs, timeout, images_path, labels_path)


def run_tflite_int8(model_path, batch_size, num_of_runs, timeout, images_path, labels_path):

    def run_single_pass(tflite_runner, imagenet):
        shape = (224, 224)
        tflite_runner.set_input_tensor(tflite_runner.input_details[0]['index'], imagenet.get_input_array(shape))
        tflite_runner.run()
        output_tensor = tflite_runner.get_output_tensor(tflite_runner.output_details[0]['index'])
        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output_tensor[i]),
                imagenet.extract_top5(output_tensor[i])
            )

    dataset = ImageNet(batch_size, "RGB", images_path, labels_path,
                       pre_processing="VGG", is1001classes=False)
    runner = TFLiteRunner(model_path)

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def main():
    args = parse_args()
    if args.framework == "tf":
        if args.model_path is None:
            raise ModelPathUnspecified(args.model_path)
        if args.precision == "fp32":
            run_tf_fp32(
                args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.labels_path
            )
        elif args.precision == "fp16":
            run_tf_fp16(
                args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.labels_path
            )
        elif args.precision == "int8":
            run_tflite_int8(
                args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.labels_path
            )
        else:
            raise UnsupportedPrecisionValueError(args.precision)

    elif args.framework == "pytorch":
        if args.precision == "fp32":
            run_pytorch_fp32(
                args.batch_size, args.num_runs, args.timeout, args.images_path, args.labels_path
            )
        else:
            raise UnsupportedPrecisionValueError(args.precision)
    else:
        raise FrameworkUnsupportedError(args.framework)


if __name__ == "__main__":
    main()
