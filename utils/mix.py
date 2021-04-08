import os
import tensorflow as tf


def batch(iterable, n=1):
    length = len(iterable)
    for ndx in range(0, length, n):
        yield iterable[ndx:min(ndx + n, length)]


def calculate_images():
    path_to_images = os.environ['IMAGES_PATH']
    number_of_images = len(os.listdir(path_to_images))
    return number_of_images


def last_5chars(x):
    return x[-10:-5]


def initialize_graph(model_path, input_tensor_name, output_tensor_name):
    with tf.io.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph, graph.get_tensor_by_name(input_tensor_name), graph.get_tensor_by_name(output_tensor_name)