import argparse
import tensorflow as tf
from utils.graphloader import load_graph
from utils.imagenet import ImageNet
from utils.vggpreprocessor import vgg_preprocessor
resnet50_fp32 = "../models/resnet50_v1.5_fp32.pb"


def parse_args():
    parser = argparse.ArgumentParser(description='benchmark resnet model')

    parser.add_argument('-m', '--model_path', type=str, required=False, help='Path to frozen model directory.')
    parser.add_argument('-t', '--timeout', type=str, required=False, help='Timeout for evaluation dataset processing')
    return parser.parse_args()


def benchmark():
    graph = load_graph(resnet50_fp32)
    input_tensor = graph.get_tensor_by_name("input_tensor:0")
    output_tensor = graph.get_tensor_by_name("softmax_tensor:0")
    sess = tf.compat.v1.Session(graph=graph)

    image_net = ImageNet()
    our_input = image_net.get_input_tensor(1, (224, 224), vgg_preprocessor)
#
#
# start = time.time()
# result = sess.run(output_tensor, feed_dict={input_tensor: preprocessed_image})
# end = time.time()
# total_time += (end - start)


def main():
    args = parse_args()
    benchmark()


if __name__ == "__main__":
    print (__name__)
    main()
