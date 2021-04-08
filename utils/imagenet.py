import sys
import os.path
import time
from os import path
import numpy as np
import cv2
from utils.mix import batch, last_5chars, initialize_graph
import tensorflow as tf


class ImageNet:

    def __init__(self, model_path, batch_size, input_tensor_name, output_tensor_name, is1001classes):

        # paths
        self.image_label = '/model_zoo/utils/val.txt'
        self.images_path = os.environ['IMAGES_PATH']

        # images
        self.parent_list = os.listdir(self.images_path)
        self.parent_list_sorted = sorted(self.parent_list, key=last_5chars)
        self.g = batch(self.parent_list_sorted, batch_size)

        # tensorflow session
        self.graph, self.input_tensor, self.output_tensor = initialize_graph(model_path,
                                                                             input_tensor_name,
                                                                             output_tensor_name)
        self.sess = tf.compat.v1.Session(graph=self.graph)
        self.labels_iterator = self.get_labels_iterator(is1001classes)

        # Benchmarks
        self.image_count = 0
        self.total_time = 0.0
        self.top_1 = 0
        self.top_5 = 0

        if not path.exists(self.images_path):
            print("path doesn't exist")
            sys.exit(1)
        else:
            print('works')

    def get_input_tensor(self, input_shape, preprocess, model):
        final_batch = np.empty((0, 224, 224, 3))

        for i in self.g.__next__():
            img = cv2.imread(os.path.join(self.images_path, i))
            resized_img = cv2.resize(img, input_shape)
            preprocessed_img = preprocess(resized_img, model)
            final_batch = np.append(final_batch, preprocessed_img, axis=0)

        print(final_batch.shape)
        return final_batch

    def run(self, preprocessed_input):
        start = time.time()
        result = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: preprocessed_input})
        end = time.time()
        self.total_time += (end - start)

        top_1_index = np.where(result == np.max(result))[1]
        top_1_index = int(top_1_index)
        label = next(self.labels_iterator)

        result_flattened = result.flatten()
        top_5_indices = result_flattened.argsort()[-5:][::-1]

        self.image_count += 1

        if label == top_1_index:
            self.top_1 += 1
            self.top_5 += 1
        elif label in top_5_indices:
            self.top_5 += 1

        return result

    def print_benchmarks(self):

        print('------------------------------')

        top_1_accuracy = ((self.top_1 / self.image_count) * 100)
        print("top-1 accuracy: %.2f" % top_1_accuracy, "%")

        print('------------------------------')

        top_5_accuracy = ((self.top_5 / self.image_count) * 100)
        print("top-5 accuracy: %.2f" % top_5_accuracy, "%")

        print('------------------------------')

        minutes = self.total_time / 60
        print("total time of run inferences: %.2f" % minutes, "Minutes")

        print('------------------------------')

        latency_in_milliseconds = (self.total_time / self.image_count) * 1000
        print("average latency in miliseconds: %.4f" % latency_in_milliseconds)

        print('------------------------------')

        latency_in_fps = self.image_count / self.total_time
        print("average latency in fps: %.4f" % latency_in_fps)

        print('------------------------------')

    def get_labels_iterator(self, is1001classes=False):
        file = open(self.image_label, 'r')
        lines = file.readlines()
        labels = []
        for line in lines:
            label = int(line[28:])
            label_plus_one = label + 1
            if is1001classes:
                labels.append(label_plus_one)
            else:
                labels.append(label)
        labels_iterator = iter(labels)

        return labels_iterator
