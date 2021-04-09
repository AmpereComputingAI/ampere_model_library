import argparse
import time

from utils.imagenet import ImageNet
from utils.mix import vgg_preprocessor
from utils.tf_utils import TFFrozenModelRunner


def parse_args():
    parser = argparse.ArgumentParser(description='benchmark resnet model')

    parser.add_argument('-m', '--model_path', type=str, required=False, help='Path to frozen model directory.')
    parser.add_argument('-t', '--timeout_in_minutes', type=str, required=False, help='timeout for processing evaluation'
                                                                                     'dataset. Default is 1 minute.')
    parser.add_argument('-b', '--batch_size', type=int, required=False, help='batch size for processing evaluation'
                                                                             'dataset.')
    parser.add_argument('-p', '--precision', type=int, required=False, help='specify the precision of the model.'
                                                                            'Allowed precisions: int8, fp16, fp32')
    return parser.parse_args()


def benchmark(model_path, batch_size, timeout_in_minutes=1):

    # Initialize ImageNet class
    image_net = ImageNet(batch_size, True, 'BGR')

    # TF runner initialization
    tf_runner = TFFrozenModelRunner(model_path, ['softmax_tensor:0'])

    check = 0

    # timeout
    timeout = time.time() + 60 * float(timeout_in_minutes)
    for n in range(0, image_net.number_of_iterations):
        if time.time() > timeout:
            break

        # preprocess input
        preprocessed_input = image_net.get_input_tensor((224, 224), vgg_preprocessor)

        # set input tensor and run interference
        tf_runner.set_input_tensor('input_tensor:0', preprocessed_input)
        result = tf_runner.run()

        # record measurement
        image_net.record_measurement(result)

        check += 1
        print(check)

    # print benchmarks and accuracy
    tf_runner.print_performance_metrics(batch_size)
    image_net.print_accuracy()


def main():
    args = parse_args()
    benchmark(args.model_path, args.batch_size, args.timeout_in_minutes)


if __name__ == "__main__":
    main()
