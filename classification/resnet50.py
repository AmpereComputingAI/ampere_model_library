import argparse
import time

import tensorflow as tf
import numpy as np

from utils.graphloader import load_graph
from utils.imagenet import ImageNet
from utils.vggpreprocessor import vgg_preprocessor
from utils.mix import calculate_images


labels = '/model_zoo/utils/ILSVRC2012_validation_ground_truth.txt'


def parse_args():
    parser = argparse.ArgumentParser(description='benchmark resnet model')

    parser.add_argument('-m', '--model_path', type=str, required=False, help='Path to frozen model directory.')
    parser.add_argument('-t', '--timeout_in_minutes', type=str, required=False, help='timeout for processing evaluation'
                                                                                     'dataset. Default is 1 minute.')
    parser.add_argument('-b', '--batch_size', type=int, required=False, help='batch size for processing evaluation'
                                                                             'dataset.')
    return parser.parse_args()


def benchmark(model_path, batch_size, timeout_in_minutes=1):

    # Read labels
    file_with_labels = open(labels, 'r')
    lines = file_with_labels.readlines()
    labels_iterator = iter(lines)

    # loading graph and starting tensorflow session
    graph = load_graph(model_path)
    input_tensor = graph.get_tensor_by_name("input_tensor:0")
    output_tensor = graph.get_tensor_by_name("softmax_tensor:0")
    sess = tf.compat.v1.Session(graph=graph)

    number_of_images = calculate_images()
    num_of_iter = number_of_images/batch_size
    image_net = ImageNet(batch_size)

    # benchmarking
    total_time = 0.0
    correct_count = 0
    incorrect_count = 0
    image_count = 0

    # timeout
    timeout = time.time() + 60 * int(timeout_in_minutes)
    for n in range(0, int(num_of_iter)):
        if time.time() > timeout:
            break

        image_count += 1
        path, preprocessed_input = image_net.get_input_tensor((224, 224), vgg_preprocessor)

        print(path)

        start = time.time()
        result = sess.run(output_tensor, feed_dict={input_tensor: preprocessed_input})
        end = time.time()
        total_time += (end - start)

        max_value_index = np.where(result == np.max(result))[1]
        max_value_index = int(max_value_index)
        print(max_value_index)

        if max_value_index == next(labels_iterator):
            correct_count += 1
        else:
            incorrect_count += 1

    end = time.time()
    total_time += (end - start)

    print('------------------------------')

    accuracy = ((correct_count / image_count) * 100)
    print("top-1 accuracy: %.2f" % accuracy, "%")

    print('------------------------------')

    minutes = total_time / 60
    print("total time of run inferences: %.2f" % minutes, "Minutes")

    print('------------------------------')

    latency_in_miliseconds = (total_time / image_count) * 1000
    print("average latency in miliseconds: %.4f" % latency_in_miliseconds)

    print('------------------------------')

    latency_in_fps = image_count / total_time
    print("average latency in fps: %.4f" % latency_in_fps)

    print('------------------------------')


def main():
    args = parse_args()
    benchmark(args.model_path, args.batch_size, args.timeout_in_minutes)


if __name__ == "__main__":
    main()
