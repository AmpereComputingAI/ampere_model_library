import time
from tqdm.auto import tqdm
import argparse
from utils.imagenet import ImageNet
import utils.misc as utils
from utils.tf import TFFrozenModelRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Run ResNet-50 v1.5 model.")
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
                        type=int, default=-1,
                        help="number of passes through network to execute")
    parser.add_argument("--images_path",
                        type=str,
                        help="path to directory with ImageNet validation images")
    parser.add_argument("--labels_path",
                        type=str,
                        help="path to file with validation labels")
    return parser.parse_args()


def run_tf_fp32(model_path, batch_size, num_of_runs, timeout, images_path, labels_path):
    def run_single_pass():
        runner.set_input_tensor("input_tensor:0", imagenet.get_input_array(shape))
        output = runner.run()
        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output["softmax_tensor:0"][i]),
                imagenet.extract_top5(output["softmax_tensor:0"][i])
            )

    shape = (224, 224)
    imagenet = ImageNet(batch_size, "RGB", images_path, labels_path,
                        pre_processing_approach="VGG", is1001classes=True)

    if imagenet.available_images_count < num_of_runs:
        utils.print_goodbye_message_and_die(
            f"Number of runs requested exceeds number of images available in dataset!")

    runner = TFFrozenModelRunner(model_path, ["softmax_tensor:0"])

    try:
        if num_of_runs == -1:
            start = time.time()
            while time.time() - start < timeout:
                run_single_pass()
        else:
            for _ in tqdm(range(num_of_runs)):
                run_single_pass()
    except imagenet.OutOfImageNetImages:
        pass

    acc = imagenet.summarize_accuracy()
    perf = runner.print_performance_metrics(batch_size)
    return acc, perf


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
