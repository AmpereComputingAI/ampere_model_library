import sys
import os.path
from os import path
import numpy as np
import cv2
from utils.mix import batch, last_5chars
import tensorflow as tf
image_label = '/model_zoo/utils/val.txt'


def initialize_graph(model_path, input_tensor_name, output_tensor_name):
    with tf.io.gfile.GFile(model_path, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph, graph.get_tensor_by_name("input_tensor:0"), graph.get_tensor_by_name("softmax_tensor:0")


class ImageNet:

    def __init__(self, model_path, batch_size, input_tensor_name, output_tensor_name):

        self.images_path = os.environ['IMAGES_PATH']
        self.number_of_images = 50000
        self.image_count = 0
        self.parent_list = os.listdir(self.images_path)
        self.parent_list_sorted = sorted(self.parent_list, key=last_5chars)
        self.g = batch(self.parent_list_sorted, batch_size)
        self.graph, self.input_tensor, self.output_tensor = initialize_graph(model_path)
        self.sess = tf.compat.v1.Session(graph=self.graph)

        if not path.exists(self.images_path):
            print("path doesn't exist")
            sys.exit(1)
        else:
            print('works')

    def get_input_tensor(self, input_shape, preprocess):
        final_batch = np.empty((0, 224, 224, 3))

        for i in self.g.__next__():
            img_path = os.path.join(self.images_path, i)
            img = cv2.imread(os.path.join(self.images_path, i))
            resized_img = cv2.resize(img, input_shape)
            preprocessed_img = preprocess(resized_img)
            final_batch = np.append(final_batch, preprocessed_img, axis=0)

        return img_path, final_batch

    def get_labels_iterator(self):
        file = open(image_label, 'r')
        lines = file.readlines()
        labels = []
        for line in lines:
            label = int(line[28:])
            label_plus_one = label + 1
            labels.append(label_plus_one)

        return iter(labels)
