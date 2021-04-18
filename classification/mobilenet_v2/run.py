import time
import argparse
from utils.imagenet import ImageNet
from utils.tf import TFFrozenModelRunner
from utils.pre_processors import pre_process_inception


def parse_args():
    parser = argparse.ArgumentParser(description="Run MobileNet v2 model.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=True,
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
    return parser.parse_args()


def run_tf_fp32(model_path, batch_size, num_of_runs, timeout, images_path, labels_path):
    shape = (224, 224)
    imagenet = ImageNet(batch_size, "RGB", images_path, labels_path,
                        pre_processing_func=pre_process_inception, is1001classes=True)

    runner = TFFrozenModelRunner(model_path, ["MobilenetV2/Predictions/Reshape_1:0"])

    iter = 0
    start = time.time()
    while True:
        if num_of_runs is None:
            if time.time() - start > timeout:
                break
        elif not iter < num_of_runs:
            break

        try:
            runner.set_input_tensor("input:0", imagenet.get_input_array(shape))
        except imagenet.OutOfImageNetImages:
            break

        output = runner.run()
        iter += 1

        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output["MobilenetV2/Predictions/Reshape_1:0"][i]),
                imagenet.extract_top5(output["MobilenetV2/Predictions/Reshape_1:0"][i])
            )

    imagenet.summarize_accuracy()
    runner.print_performance_metrics(batch_size)


def main():
    args = parse_args()
    if args.precision == "fp32":
        run_tf_fp32(
            args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.labels_path
        )
    else:
        assert False


if __name__ == "__main__":
    main()
