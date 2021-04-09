import sys
import os.path
from os import path
import numpy as np
import cv2
from utils.mix import batch, last_5chars
from tensorflow.keras.preprocessing import image


class ImageNet:
    """
    A class providing facilities for preprocessing and postprocessing of ImageNet validation dataset.
    """

    def __init__(self, batch_size, is1001classes, channels):

        """
        A function for initialization of the class by providing batch size, a boolean whether a model has 1000 classes
        or 1001 classes and what color channels it uses.

        :param batch_size: int, a batch (set of images) on which a model wil run an inference,
        usual sizes are 1, 4, 16, 32, 64
        :param is1001classes: boolean, True if model has 10001 classes ( one extra class for background ) &
        False if it doesn't
        :param channels: str, specifies what color channels a model accepts, eg. "BGR", "RGB"
        """

        # paths
        self.image_label = '/model_zoo/utils/val.txt'
        self.images_path = os.environ['IMAGES_PATH']

        # images
        self.parent_list = os.listdir(self.images_path)
        self.parent_list_sorted = sorted(self.parent_list, key=last_5chars)
        self.g = batch(self.parent_list_sorted, batch_size)
        self.channels = channels
        self.number_of_images = 50000
        self.number_of_iterations = int(self.number_of_images / batch_size)

        # labels
        self.is1001classes = is1001classes
        self.labels_iterator = self.get_labels_iterator()

        # Accuracy
        self.image_count = 0
        self.top_1 = 0
        self.top_5 = 0

        if not path.exists(self.images_path):
            print("path doesn't exist")
            sys.exit(1)

    def get_input_tensor(self, input_shape, preprocess):
        """
        A function providing preprocess images in batches.

        :param input_shape: tuple, a shape of input image for the model, eg. (224, 224)
        :param preprocess: a function performing preprocessing
        :return: numpy array of images, eg. (1, 224, 224, 3)
        """
        final_batch = np.empty((0, 224, 224, 3))

        for i in self.g.__next__():

            # note: cv2 returns by default BGR
            img = cv2.imread(os.path.join(self.images_path, i))

            if self.channels == 'RGB':
                img = img[:, :, [2, 1, 0]]

            resized_img = cv2.resize(img, input_shape)
            img_array = image.img_to_array(resized_img)
            img_array_expanded_dims = np.expand_dims(img_array, axis=0)
            print(type(img_array_expanded_dims))
            print(img_array_expanded_dims.shape)
            preprocessed_img = preprocess(img_array_expanded_dims)
            final_batch = np.append(final_batch, preprocessed_img, axis=0)

        return final_batch

    def record_measurement(self, result):
        """
        A function recording measurements of each run inference.

        :param result: dictionary, a dictionary of output tensors and it's results
        """
        # Get value of first key in the dictionary
        result = result.get(list(result.keys())[0])

        # get index of highest value from array of results
        top_1_index = int(np.where(result == np.max(result))[1])

        # create sorted array of indices of 5 highest values
        top_5_indices = result.flatten().argsort()[-5:][::-1]

        label = next(self.labels_iterator)
        self.image_count += 1

        if label == top_1_index:
            self.top_1 += 1
            self.top_5 += 1
        elif label in top_5_indices:
            self.top_5 += 1

    def print_accuracy(self):
        """
        A function printing accuracy obtained after running all batches of images.
        """
        top_1_accuracy = ((self.top_1 / self.image_count) * 100)
        print("top-1 accuracy: %.2f" % top_1_accuracy, "%")

        top_5_accuracy = ((self.top_5 / self.image_count) * 100)
        print("top-5 accuracy: %.2f" % top_5_accuracy, "%")

    def get_labels_iterator(self):
        """
        A function which creates an iterator of ground truth labels corresponding to each image.

        :return: iterator, returns labels iterator
        """
        file = open(self.image_label, 'r')
        lines = file.readlines()
        labels = []
        for line in lines:
            label = int(line[28:])
            label_plus_one = label + 1
            if self.is1001classes:
                labels.append(label_plus_one)
            else:
                labels.append(label)
        labels_iterator = iter(labels)

        return labels_iterator
