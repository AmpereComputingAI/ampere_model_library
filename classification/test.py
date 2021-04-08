import argparse
import time
import sys

import tensorflow as tf
import numpy as np
# np.set_printoptions(threshold=sys.maxsize)

from utils.graphloader import load_graph
from utils.imagenet import ImageNet
from utils.vggpreprocessor import vgg_preprocessor
from utils.mix import calculate_images


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
    image_net = ImageNet(model_path, batch_size, "input_tensor:0", "softmax_tensor:0")

    # Read labels
    labels_iterator = image_net.get_labels_iterator()

    # loading graph and starting tensorflow session
    graph = load_graph(model_path)
    input_tensor = graph.get_tensor_by_name("input_tensor:0")
    output_tensor = graph.get_tensor_by_name("softmax_tensor:0")
    sess = tf.compat.v1.Session(graph=graph)

    number_of_images = calculate_images()
    num_of_iter = number_of_images/batch_size

    # benchmarking
    total_time = 0.0
    top_1 = 0
    top_5 = 0
    incorrect_count = 0
    image_count = 0

    # timeout
    timeout = time.time() + 60 * float(timeout_in_minutes)
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

        result_flattened = result.flatten()
        top_5_indices = result_flattened.argsort()[-5:][::-1]

        top_1_index = np.where(result == np.max(result))[1]
        top_1_index = int(top_1_index)
        label = next(labels_iterator)

        if label == top_1_index:
            top_1 += 1
            top_5 += 1
        elif label in top_5_indices:
            top_5 += 1
        else:
            incorrect_count += 1

    end = time.time()
    total_time += (end - start)

    print('------------------------------')

    top_1_accuracy = ((top_1 / image_count) * 100)
    print("top-1 accuracy: %.2f" % top_1_accuracy, "%")

    print('------------------------------')

    top_5_accuracy = ((top_5 / image_count) * 100)
    print("top-5 accuracy: %.2f" % top_5_accuracy, "%")

    print('------------------------------')

    minutes = total_time / 60
    print("total time of run inferences: %.2f" % minutes, "Minutes")

    print('------------------------------')

    latency_in_milliseconds = (total_time / image_count) * 1000
    print("average latency in miliseconds: %.4f" % latency_in_milliseconds)

    print('------------------------------')

    latency_in_fps = image_count / total_time
    print("average latency in fps: %.4f" % latency_in_fps)

    print('------------------------------')


def main():
    args = parse_args()
    benchmark(args.model_path, args.batch_size, args.timeout_in_minutes)


if __name__ == "__main__":
    main()
