import sys
import os.path
import time
from os import path
import numpy as np
import cv2
from utils.mix import batch, last_5chars, initialize_graph
import tensorflow as tf
from tensorflow.keras.preprocessing import image


class ImageNet:

    def __init__(self, model_path, batch_size, is1001classes):

        # paths
        self.image_label = '/model_zoo/utils/val.txt'
        self.images_path = os.environ['IMAGES_PATH']

        # images
        self.parent_list = os.listdir(self.images_path)
        self.parent_list_sorted = sorted(self.parent_list, key=last_5chars)
        self.g = batch(self.parent_list_sorted, batch_size)

        # tensorflow session
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

    def get_input_tensor(self, input_shape, preprocess, channels):
        final_batch = np.empty((0, 224, 224, 3))

        for i in self.g.__next__():
            img = cv2.imread(os.path.join(self.images_path, i))

            # cv2 returns by default BGR
            if channels == 'RGB':
                new_image = img[:, :, [2, 1, 0]]
                resized_img = cv2.resize(new_image, input_shape)
            elif channels == "BGR":
                resized_img = cv2.resize(img, input_shape)

            img_array = image.img_to_array(resized_img)
            img_array_expanded_dims = np.expand_dims(img_array, axis=0)
            preprocessed_img = preprocess(img_array_expanded_dims)
            print(preprocessed_img.shape)
            final_batch = np.append(final_batch, preprocessed_img, axis=0)

        print(final_batch.shape)
        return final_batch

    def perform_measurement(self, result):
        # start = time.time()
        # result = self.sess.run(self.output_tensor, feed_dict={self.input_tensor: preprocessed_input})
        # end = time.time()
        # self.total_time += (end - start)
        #
        # print(result.shape)

        result = result.get('softmax_tensor:0')

        top_1_index = np.where(result == np.max(result))[1]
        top_1_index = int(top_1_index)
        print("top1")
        print(top_1_index)
        label = next(self.labels_iterator)
        print("label")
        print(label)

        result_flattened = result.flatten()
        top_5_indices = result_flattened.argsort()[-5:][::-1]
        print('top5')
        print(top_5_indices)

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

        # print('------------------------------')

        # minutes = self.total_time / 60
        # print("total time of run inferences: %.2f" % minutes, "Minutes")
        #
        # print('------------------------------')
        #
        # latency_in_milliseconds = (self.total_time / self.image_count) * 1000
        # print("average latency in miliseconds: %.4f" % latency_in_milliseconds)
        #
        # print('------------------------------')
        #
        # latency_in_fps = self.image_count / self.total_time
        # print("average latency in fps: %.4f" % latency_in_fps)

        # print('------------------------------')

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
