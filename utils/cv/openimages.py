# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import json
import pathlib
import utils.misc as utils
from utils.cv.coco import COCOBaseDataset


class OpenImagesDataset(COCOBaseDataset):
    """
    A class providing facilities to measure accuracy of object detection models trained on OpenImages dataset.
    """

    def __init__(self,
                 batch_size: int, color_model: str,
                 images_path=None, annotations_path=None, pre_processing=None, sort_ascending=False, order="NHWC"):
        """
        A function initializing the class.
        :param batch_size: int, size of batch intended to be processed
        :param images_path: str, path to directory containing OpenImages images
        :param annotations_path: str, path to file containing OpenImages annotations
        :param pre_processing: pre-processing approach to be applied
        :param sort_ascending: bool, parameter setting whether images in dataset should be processed in ascending order
        regarding their files' names
        """

        if images_path is None:
            env_var = "OPENIMAGES_IMG_PATH"
            images_path = utils.get_env_variable(
                env_var, f"Path to OpenImages images directory has not been specified with {env_var} flag")
        if annotations_path is None:
            env_var = "OPENIMAGES_ANNO_PATH"
            annotations_path = utils.get_env_variable(
                env_var, f"Path to OpenImages annotations file has not been specified with {env_var} flag")

        self.__images_path = images_path
        self.__annotations_path = annotations_path
        self.__file_names = self.__parse_annotations_file(self.__annotations_path)
        super().__init__(batch_size, color_model, annotations_path, pre_processing, order, sort_ascending)
    
    def __parse_annotations_file(self, annotations_path):
        """
        A function parsing annotation file for OpenImages validation dataset.

        :param annotations_path: str, path to file containing annotations
        :return: list of strings
        """

        images = {}
        with open(annotations_path, "r") as f:
            openimages = json.load(f)
        for i in openimages["images"]:
            images[i["id"]] = {"file_name": i["file_name"],
                               "height": i["height"],
                               "width": i["width"],
                               "bbox": [],
                               "category": []}
        for a in openimages["annotations"]:
            i = images.get(a["image_id"])
            if i is None:
                continue
            catagory_ids = a.get("category_id")
            i["category"].append(catagory_ids)
            i["bbox"].append(a.get("bbox"))
        return images

    def _get_path_to_img(self):
        """
        A function providing path to the OpenImages image.
        :return: pathlib.PurePath object containing path to the image
        """
        try:
            image_id = self._image_ids[self._current_img]
        except IndexError:
            raise utils.OutOfInstances("No more OpenImages images to process in the directory provided")
        self._current_image_ids.append(image_id)
        image_path = self.__file_names[image_id]["file_name"]
        self._current_img += 1
        return pathlib.PurePath(self.__images_path, image_path)
