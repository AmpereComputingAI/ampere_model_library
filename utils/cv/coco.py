# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import pathlib
import utils.misc as utils
from utils.cv.object_detection import ObjectDetectionDataset


class COCODataset(ObjectDetectionDataset):
    """
    A class providing facilities to measure accuracy of object detection models trained on COCO dataset.
    """

    def __init__(self,
                 batch_size: int, color_model: str, images_filename_base: str,
                 images_path=None, annotations_path=None, pre_processing=None, sort_ascending=False, order="NHWC"):
        """
        A function initializing the class.
        :param batch_size: int, size of batch intended to be processed
        :param images_filename_base: str, default name of image files in given COCO dataset,
        eg. "COCO_val2014_000000000000"
        :param images_path: str, path to directory containing COCO images
        :param annotations_path: str, path to file containing COCO annotations
        :param pre_processing: pre-processing approach to be applied
        :param sort_ascending: bool, parameter setting whether images in dataset should be processed in ascending order
        regarding their files' names
        """

        if images_path is None:
            env_var = "COCO_IMG_PATH"
            images_path = utils.get_env_variable(
                env_var, f"Path to COCO images directory has not been specified with {env_var} flag")
        if annotations_path is None:
            env_var = "COCO_ANNO_PATH"
            annotations_path = utils.get_env_variable(
                env_var, f"Path to COCO annotations file has not been specified with {env_var} flag")

        self.__images_filename_base = images_filename_base
        self.__images_filename_ext = ".jpg"
        self.__images_path = images_path
        super().__init__(batch_size, color_model, annotations_path, pre_processing, order, sort_ascending)

    def _get_path_to_img(self):
        """
        A function providing path to the COCO image.
        :return: pathlib.PurePath object containing path to the image
        """
        try:
            image_id = self._image_ids[self._current_img]
        except IndexError:
            raise utils.OutOfInstances("No more COCO images to process in the directory provided")
        self._current_image_ids.append(image_id)
        image_path = self.__images_filename_base[:-len(str(image_id))] + str(image_id) + self.__images_filename_ext
        self._current_img += 1
        return pathlib.PurePath(self.__images_path, image_path)
