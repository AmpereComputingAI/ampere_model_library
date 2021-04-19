import cv2
import numpy as np
import utils.misc as utils


class OutOfInstances(Exception):
    """
    An exception class being raised as an error in case of lack of further images to process by the pipeline.
    """
    pass


class ImageDataset:
    """
    A class providing facilities shared by image-related datasets' pipelines like ImageNet and COCO.
    """
    def __init__(self):
        pass

    def __resize_image(self, image_array, target_shape):
        """
        A function resizing an image.

        :param image_array: numpy array with image data
        :param target_shape: tuple of intended image shape (height, width)
        :return: numpy array with resized image data, tuple with ratios of resizing applied (height, width)
        """
        vertical_ratio = target_shape[0] / image_array.shape[0]
        horizontal_ratio = target_shape[1] / image_array.shape[1]
        return cv2.resize(image_array, target_shape), (vertical_ratio, horizontal_ratio)

    def __load_image(self, image_path, target_shape, color_model: str):
        """
        A function loading image available under the supplied path and then applying proper rescaling/resizing.

        :param image_path: PathLib.PurePath or str, path to the image
        :param target_shape: tuple of intended image shape (height, width)
        :param color_model: str, color model of image data, possible values: ["RGB", "BGR"]
        :return: numpy array with resized image data in NHWC/NCHW format
        """
        if color_model not in ["RGB", "BGR"]:
            utils.print_goodbye_message_and_die(f"Color model {color_model} is not supported.")

        image_array = cv2.imread(str(image_path))
        if image_array is None:
            utils.print_goodbye_message_and_die(f"Image not found under path {str(image_path)}!")
        if color_model == "RGB":
            image_array = image_array[:, :, [2, 1, 0]]  # cv2 loads images in BGR order
        image_array, resize_ratios = self.__resize_image(image_array, target_shape)
        return np.expand_dims(image_array, axis=0), resize_ratios
