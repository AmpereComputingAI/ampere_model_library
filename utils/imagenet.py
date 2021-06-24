import numpy as np
import pathlib
import utils.misc as utils
import utils.pre_processing as pp
import utils.dataset as utils_ds


class ImageNet(utils_ds.ImageDataset):
    """
    A class providing facilities for preprocessing and postprocessing of ImageNet validation dataset.
    """

    def __init__(self, batch_size: int, color_model: str,
                 images_path=None, labels_path=None, pre_processing=None, is1001classes=False):

        if images_path is None:
            env_var = "IMAGENET_IMG_PATH"
            images_path = utils.get_env_variable(
                env_var, f"Path to ImageNet images directory has not been specified with {env_var} flag")
        if labels_path is None:
            env_var = "IMAGENET_LABELS_PATH"
            labels_path = utils.get_env_variable(
                env_var, f"Path to ImageNet labels file has not been specified with {env_var} flag")

        self.__batch_size = batch_size
        self.__color_model = color_model
        self.__images_filename_extension = ".JPEG"
        self.__images_path = images_path
        self.__pre_processing = pre_processing
        self.__current_img = 0
        self.__file_names, self.__labels = utils.parse_val_file(labels_path, is1001classes, 28, False)
        self.available_instances = len(self.__file_names)
        self.__top_1_count = 0
        self.__top_5_count = 0
        self.path_to_latest_image = None
        super().__init__()

    def __get_path_to_img(self):
        """
        A function providing path to the ImageNet image.

        :return: pathlib.PurePath object containing path to the image
        """
        try:
            file_name = self.__file_names[self.__current_img]
        except IndexError:
            raise utils_ds.OutOfInstances("No more ImageNet images to process in the directory provided")
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
            self.path_to_latest_image = self.__get_path_to_img()
            input_array[i], _ = self._ImageDataset__load_image(
                self.path_to_latest_image, target_shape, self.__color_model
            )
        if self.__pre_processing:
            input_array = pp.pre_process(input_array, self.__pre_processing, self.__color_model)
        return input_array

    def extract_top1(self, output_array):
        """
        A helper function for extracting top-1 prediction from an output array holding soft-maxed data on 1 image.

        :param output_array: 1-D numpy array containing soft-maxed logits referring to 1 image
        :return: int, index of highest value in the supplied array
        """
        top_1_index = np.argmax(output_array)
        return top_1_index

    def extract_top5(self, output_array):
        """
        A helper function for extracting top-5 predictions from an output array holding soft-maxed data on 1 image.

        :param output_array: 1-D numpy array containing soft-maxed logits referring to 1 image
        :return: list of ints, list containing indices of 5 highest values in the supplied array
        """
        top_5_indices = np.argpartition(output_array, -5)[-5:]
        return top_5_indices

    def submit_predictions(self, id_in_batch: int, top_1_index: int, top_5_indices: list):
        """
        A function meant for submitting a class predictions for a given image.

        :param id_in_batch: int, id of an image in the currently processed batch that the provided predictions relate to
        :param top_1_index: int, index of a prediction with highest confidence
        :param top_5_indices: list of ints, indices of 5 predictions with highest confidence
        :return:
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
        print("\n Top-1 accuracy = {:.3f}".format(top_1_accuracy))

        top_5_accuracy = self.__top_5_count / self.__current_img
        print(" Top-5 accuracy = {:.3f}".format(top_5_accuracy))

        print(f"\nAccuracy figures above calculated on the basis of {self.__current_img} images.")
        return {"top_1_acc": top_1_accuracy, "top_5_acc": top_5_accuracy}
