import argparse
import time

from utils.imagenet import ImageNet
from utils.misc import vgg_preprocessor
from utils.tf_utils import TFFrozenModelRunner


def parse_args():
    parser = argparse.ArgumentParser(description='benchmark resnet model')

    parser.add_argument('-m', '--model_path', type=str, required=True, help='Path to frozen model.')
    parser.add_argument('-t', '--timeout', type=int, default=60, help='Timeout for processing evaluation'
                                                                      'dataset in seconds. Default is 60 seconds.')
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='Batch size for processing evaluation'
                                                                        'dataset. Possible values: 1, 4, 16, 32, 64.'
                                                                        'Default value is 1.')
    parser.add_argument('-p', '--precision', type=str, choices=['fp32', 'fp16', 'int8'], help='Specify the precision of'
                                                                                              'the model provided with'
                                                                                              ' --model_path argument.')
    parser.add_argument('-i', '--images_path', type=str, help="Path to ImageNet validation dataset.")
    parser.add_argument('-l', '--labels_path', type=str, help="Path to validation labels.")
    return parser.parse_args()


def benchmark(model_path, batch_size, images_path, labels_path, timeout):

    # Initialize ImageNet class
    image_net = ImageNet(batch_size, True, 'BGR', images_path, labels_path)

    # TF runner initialization
    tf_runner = TFFrozenModelRunner(model_path, ['softmax_tensor:0'])

    # timeout
    time_of_start = time.time()
    count = 0

    while time.time() - time_of_start < timeout:

        # preprocess input
        preprocessed_input = image_net.get_input_tensor((224, 224), vgg_preprocessor)
        count += 1
        print(count)

        # set input tensor and run interference
        tf_runner.set_input_tensor('input_tensor:0', preprocessed_input)
        result = tf_runner.run()

        # record measurement
        image_net.record_measurement(result['softmax_tensor:0'])

    # print benchmarks and accuracy
    tf_runner.print_performance_metrics(batch_size)
    image_net.print_accuracy()


def main():
    args = parse_args()
    benchmark(args.model_path, args.batch_size, args.images_path, args.labels_path, args.timeout)


if __name__ == "__main__":
    main()
