import os
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


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


def vgg_preprocessor(image_sample, model):
    img_array = image.img_to_array(image_sample)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    if model == 'resnet':
        result = tf.keras.applications.resnet.preprocess_input(img_array_expanded_dims)
    elif model == 'mobilenet':
        result = tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

    return result
