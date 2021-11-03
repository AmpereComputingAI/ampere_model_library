import argparse
from utils.cv.imagenet import ImageNet
from utils.pytorch import PyTorchRunner
from utils.benchmark import run_model
from utils.misc import UnsupportedPrecisionValueError, FrameworkUnsupportedError

PYTORCH_MODEL_NAME = 'mobilenet_v3_large'


def parse_args():
    parser = argparse.ArgumentParser(description="Run Mobilenet V3 Large model.")
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
                        choices=["pytorch"], default="tf",
                        help="specify the framework in which a model should be run")
    return parser.parse_args()


def run_pytorch_fp(batch_size, num_of_runs, timeout, images_path, labels_path):

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
                       pre_processing='PyTorch', is1001classes=False, order='NCHW')
    runner = PyTorchRunner(PYTORCH_MODEL_NAME)

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def run_pytorch_fp32(batch_size, num_of_runs, timeout, images_path, labels_path):
    return run_pytorch_fp(batch_size, num_of_runs, timeout, images_path, labels_path)


def main():
    args = parse_args()
    if args.framework == "tf":
        raise FrameworkUnsupportedError(args.framework)

    elif args.framework == "pytorch":
        if args.precision == "fp32":
            run_pytorch_fp32(
                args.batch_size, args.num_runs, args.timeout, args.images_path, args.labels_path
            )
        else:
            raise UnsupportedPrecisionValueError(args.precision)


if __name__ == "__main__":
    main()
