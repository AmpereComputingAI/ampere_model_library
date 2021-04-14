import os.path
import numpy as np
import cv2
import utils.misc as utils
from tensorflow.keras.preprocessing import image


def get_labels_path(labels_path):
    """
    A function returning the path to file with validation dataset labels

    :return: labels_path: str
    # """
    if labels_path is None:
        try:
            labels_path = os.environ['LABELS_PATH']
        except Exception as e:
            print(e)
        finally:
            if labels_path is None:
                utils.print_goodbye_message_and_die('The path to labels was not defined!')
    return labels_path


def get_images_path(images_path):
    """
    A function returning the path to file with validation dataset labels

    :return: labels_path: str
    # """
    if images_path is None:
        try:
            images_path = os.environ['IMAGES_PATH']
        except Exception as e:
            print(e)
        finally:
            if images_path is None:
                utils.print_goodbye_message_and_die('The path to images was not defined!')
    return images_path


class ImageNet:
    """
    A class providing facilities for preprocessing and postprocessing of ImageNet validation dataset.
    """

    def __init__(self, batch_size, is1001classes, channels, images_path, labels_path):

        """
        A function for initialization of the class

        :param batch_size: int, a batch size on which a model will run an inference,
        usual sizes are 1, 4, 16, 32, 64
        :param is1001classes: boolean, True if model has 10001 classes ( one extra class for background ) &
        False if it doesn't
        :param channels: str, specifies the order of color channels a model accepts, eg. "BGR", "RGB"
        :param images_path: str, specify a path of images folder
        :param labels_path: str, specify a path of file with validation labels
        """

        # paths
        self.images_path = get_images_path(images_path)
        self.labels_path = get_labels_path(labels_path)

        # images
        self.channels = channels
        self.batch_size = batch_size

        # labels
        self.is1001classes = is1001classes
        self.labels, self.lines = self.get_labels_iterator()

        self.labels_iterator = utils.batch(self.labels, batch_size)
        self.g = utils.batch(self.lines, batch_size)

        # Accuracy
        self.image_count = 0
        self.top_1 = 0
        self.top_5 = 0

    def get_input_tensor(self, input_shape, preprocess):
        """
        A function providing preprocess images in batches.

        :param input_shape: tuple, a shape of input image for the model, eg. (224, 224)
        :param preprocess: a function performing preprocessing
        :return: numpy array of images, eg. (1, 224, 224, 3)
        """
        final_batch = np.empty((0, 224, 224, 3))

        for i in self.g.__next__():

            try:
                # note: cv2 returns by default BGR
                img = cv2.imread(os.path.join(self.images_path, i[:28]))
                assert img is not None
            except Exception as e:
                print(e)
            else:
                if self.channels == 'RGB':
                    img = img[:, :, [2, 1, 0]]

                resized_img = cv2.resize(img, input_shape)
                img_array = image.img_to_array(resized_img)
                img_array_expanded_dims = np.expand_dims(img_array, axis=0)
                preprocessed_img = preprocess(img_array_expanded_dims)
                final_batch = np.append(final_batch, preprocessed_img, axis=0)

        return final_batch

    def record_measurement(self, result):
        """
        A function recording measurements of each run inference.

        :param result: numpy array, containing the results of inference
        """

        # get the index of top 1 prediction
        top_1_indices = np.argmax(result, axis=1)

        # get the index of top 5 predictions
        top_5_indices = np.argpartition(result, -5)[:, -5:]

        # get the array of ground truth labels
        label_array = np.array(next(self.labels_iterator))

        # count the images
        self.image_count += self.batch_size

        if self.batch_size == 1:
            if label_array == top_1_indices:
                self.top_1 += 1

        if self.batch_size > 1:
            self.top_1 += np.count_nonzero(top_1_indices == label_array)

        n = 0
        for i in label_array:
            if i in top_5_indices[n]:
                self.top_5 += 1
                n += 1

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
        try:
            file = open(self.labels_path, 'r')
        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(e)
        else:
            lines = file.readlines()
            labels = []
            for line in lines:
                label = int(line[28:])
                if self.is1001classes:
                    labels.append(label + 1)
                else:
                    labels.append(label)

        return labels, lines
