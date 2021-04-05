import argparse
import time

import tensorflow as tf
import numpy as np

from utils.graphloader import load_graph
from utils.imagenet import ImageNet
from utils.vggpreprocessor import vgg_preprocessor
from utils.mix import calculate_images


labels = 'model_zoo/utils/ILSVRC2012_validation_ground_truth.txt'


def parse_args():
    parser = argparse.ArgumentParser(description='benchmark resnet model')

    parser.add_argument('-m', '--model_path', type=str, required=False, help='Path to frozen model directory.')
    parser.add_argument('-t', '--timeout_in_minutes', type=str, required=False, help='timeout for processing evaluation'
                                                                                     'dataset. Default is 1 minute.')
    parser.add_argument('-b', '--batch_size', type=int, required=False, help='batch size for processing evaluation'
                                                                             'dataset.')
    return parser.parse_args()


def benchmark(model_path, batch_size, timeout_in_minutes=1):

    graph = load_graph(model_path)
    input_tensor = graph.get_tensor_by_name("input_tensor:0")
    output_tensor = graph.get_tensor_by_name("softmax_tensor:0")
    sess = tf.compat.v1.Session(graph=graph)

    number_of_images = calculate_images()
    num_of_iter = number_of_images/batch_size

    image_net = ImageNet()
    preprocessed_input = image_net.get_input_tensor(batch_size, (224, 224), vgg_preprocessor)

    timeout = time.time() + 60 * int(timeout_in_minutes)

    total_time = 0.0
    start = time.time()

    for n in range(0, int(num_of_iter)):
        if time.time() > timeout:
            break

        result = sess.run(output_tensor, feed_dict={input_tensor: preprocessed_input})
        print(n)

        max_value_index = np.where(result == np.max(result))[1]
        max_value_index = int(max_value_index)
        print(max_value_index)

    end = time.time()
    total_time += (end - start)


def main():
    args = parse_args()
    benchmark(args.model_path, args.batch_size, args.timeout_in_minutes)


if __name__ == "__main__":
    main()
