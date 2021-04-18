import numpy as np
import pathlib
import utils.misc as utils
from utils.dataset import ImageDataset


class ImageNet(ImageDataset):
    """
    A class providing facilities for preprocessing and postprocessing of ImageNet validation dataset.
    """

    def __init__(self, batch_size: int, color_model: str,
                 images_path=None, labels_path=None, pre_processing_func=None, is1001classes=False):

        if images_path is None:
            env_var = "IMAGENET_IMG_PATH"
            images_path = utils.get_env_variable(
                env_var, f"Path to ImageNet images directory has not been specified with {env_var} flag")
        if labels_path is None:
            env_var = "IMAGENET_LABELS_PATH"
            labels_path = utils.get_env_variable(
                env_var, f"Path to ImageNet labels file has not been specified with {env_var} flag")

        self.__batch_size = batch_size
        self.__images_filename_extension = ".JPEG"
        self.__color_model = color_model
        self.__images_path = images_path
        self.__pre_processing_func = pre_processing_func
        self.__current_img = 0
        self.__file_names, self.__labels = self.__parse_val_file(labels_path, is1001classes)
        self.available_images_count = len(self.__file_names)
        self.__top_1_count = 0
        self.__top_5_count = 0
        super().__init__()

    class OutOfImageNetImages(Exception):
        """
        An exception class being raised as an error in case of lack of further images to process by the pipeline.
        """
        pass

    def __parse_val_file(self, labels_path, is1001classes):
        if utils.get_hash_of_a_file(labels_path) != "b6284a7c08fba47457c2c1f6049a156e":
            utils.print_goodbye_message_and_die("Wrong labels file supplied!")

        boundary = 28  # single line of labels file looks like this "ILSVRC2012_val_00050000.JPEG 456"
        file = open(labels_path, "r")
        lines = file.readlines()
        file_names = list()
        labels = list()
        for line in lines:
            file_name = line[:boundary]
            file_names.append(file_name)
            label = int(line[boundary:])
            if is1001classes:
                labels.append(label + 1)
            else:
                labels.append(label)
        return file_names, labels

    def __get_path_to_img(self):
        """
        A function providing path to the ImageNet image.

        :return: pathlib.PurePath object containing path to the image
        """
        try:
            file_name = self.__file_names[self.__current_img]
        except IndexError:
            raise self.OutOfImageNetImages("No more ImageNet images to process in the directory provided")
        self.__current_img += 1
        return pathlib.PurePath(self.__images_path, file_name)

    def get_input_array(self, target_shape):
        """
        A function returning an array containing pre-processed rescaled image's or multiple images' data.

        :param target_shape: tuple of intended image shape (height, width)
        :return: numpy array containing rescaled, pre-processed image data of batch size requested at class
        initialization
        """
        input_array = np.empty([self.__batch_size, *target_shape, 3])  # NHWC order
        for i in range(self.__batch_size):
            input_array[i], _ = self._ImageDataset__load_image(
                self.__get_path_to_img(), target_shape, self.__color_model
            )
        if self.__pre_processing_func:
            input_array = self.__pre_processing_func(input_array)
        return input_array

    def extract_top1(self, output_array):
        top_1_index = np.argmax(output_array)
        return top_1_index

    def extract_top5(self, output_array):
        top_5_indices = np.argpartition(output_array, -5)[-5:]
        return top_5_indices

    def submit_predictions(self, id_in_batch: int, top_1_index: int, top_5_indices: list):
        """
        A function for submitting a class predictions for a given image.

        :param id_in_batch: int, id of an image in the currently processed batch that the provided predictions relate to
        :param top_1_prediction: int, index of top-1 prediction
        :param top_5_predictions: list of ints, indices of top-5 predictions (including top-1 supplied separately)
        """
        ground_truth = self.__labels[self.__current_img - self.__batch_size + id_in_batch]
        self.__top_1_count += int(ground_truth == top_1_index)
        self.__top_5_count += int(ground_truth in top_5_indices)

    def summarize_accuracy(self):
        """
        A function summarizing the accuracy achieved on the images obtained with get_input_array() calls on which
        predictions done where supplied with submit_predictions() function.
        """
        top_1_accuracy = self.__top_1_count / self.__current_img
        print("\n Top-1 accuracy: {:.2f}".format(top_1_accuracy))

        top_5_accuracy = self.__top_5_count / self.__current_img
        print(" Top-5 accuracy = {:.2f}\n".format(top_5_accuracy))

        print(f"\nAccuracy figures above calculated on the basis of {self.__current_img} images.")
