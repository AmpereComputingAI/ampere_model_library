# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import os
import pathlib
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import utils.misc as utils
import utils.cv.pre_processing as pp
from utils.cv.dataset import ImageDataset
from utils.misc import download_coco_dataset, download_coco_labels


class COCOBaseDataset(ImageDataset):

    def __init__(self, batch_size, color_model, annotations_path, pre_processing, order: str, sort_ascending: bool,
                 task: str = "bbox"):
        assert task in ["segm", "bbox"]
        self._task = task
        self.__batch_size = batch_size
        self.__color_model = color_model
        self.__annotations_path = annotations_path
        self.__pre_processing = pre_processing
        self.__order = order
        self._predictions = list()
        self.__current_image_ratios = list()
        self.__ground_truth = COCO(annotations_path)
        self._current_img = 0
        self._current_image_ids = list()
        self._image_ids = self.__ground_truth.getImgIds()
        if sort_ascending:
            self._image_ids = sorted(self._image_ids)
        self.available_instances = len(self._image_ids)
        self.path_to_latest_image = None
        super().__init__()

    def __reset_containers(self):
        """
        A function resetting containers (lists) containing data on the on-going batch.
        """
        self._current_image_ids = list()
        self.__current_image_ratios = list()

    def __load_image_and_store_ratios(self, target_shape):
        """
        A function loading single image and storing rescale ratios.
        :param target_shape: tuple of intended image shape (height, width)
        :return: numpy array containing rescaled image data
        """
        self.path_to_latest_image = self._get_path_to_img()
        input_array, resize_ratios = self._ImageDataset__load_image(
            self.path_to_latest_image, target_shape, self.__color_model, self.__order)
        self.__current_image_ratios.append(resize_ratios)
        return input_array

    def reset(self):
        self._current_img = 0
        self._predictions = list()
        return True

    def get_input_array(self, target_shape=None):
        """
        A function returning an array containing pre-processed rescaled image's or multiple images' data.
        :param target_shape: tuple of intended image shape (height, width)
        :return: numpy array containing rescaled, pre-processed image data of batch size requested at class
        initialization
        """
        self.__reset_containers()

        if target_shape is None:
            assert self.__batch_size == 1, "Padding not implemented, " \
                                           "only batch size == 1 compatible with native image shape"
            input_array = np.expand_dims(self.__load_image_and_store_ratios(target_shape), axis=0)
        else:
            if self.__order == 'NCHW':
                input_array = np.empty([self.__batch_size, 3, *target_shape])  # NCHW order
            else:
                input_array = np.empty([self.__batch_size, *target_shape, 3])  # NHWC order
            for i in range(self.__batch_size):
                input_array[i] = self.__load_image_and_store_ratios(target_shape)

        if self.__pre_processing:
            input_array = pp.pre_process(input_array, self.__pre_processing, self.__color_model)
        return input_array

    def convert_bbox_to_coco_order(self, bbox, left=0, top=1, right=2, bottom=3, absolute=True):
        """
        A helper function allowing for an easy switch of bbox order.
        Sometimes networks return order of bbox boundary values in different order than the default COCO's:
        left -> top -> right -> bottom
        :param bbox: list, list containing bbox coordinates
        :param left: int, index under which a left boundary is being stored
        :param top: int, index under which a top boundary is being stored
        :param right: int, index under which a right boundary / shift to the right is being stored
        :param bottom: int, index under which a bottom boundary / shift to the bottom is being stored
        :param absolute: bool, if True right/bottom coordinates are being converted to shifts
        :return: bbox with reordered and converted data
        """
        left = bbox[left]
        top = bbox[top]
        right = bbox[right]
        bottom = bbox[bottom]
        if absolute:
            right -= left
            bottom -= top
        return [left, top, right, bottom]

    def rescale_bbox(self, id_in_batch: int, bbox: list):
        """
        A function rescaling bbox coordinates back to the original scale.
        :param id_in_batch: int, id of an image in the currently processed batch that the provided bbox relates to
        :param bbox: list, a list containing coordinates of bbox already in COCO format
        :return: list, bbox in the original scale
        """

        bbox[0] /= self.__current_image_ratios[id_in_batch][1]  # left boundary divided by horizontal ratio
        bbox[1] /= self.__current_image_ratios[id_in_batch][0]  # top boundary divided by vertical ratio
        bbox[2] /= self.__current_image_ratios[id_in_batch][1]  # shift to the right divided by horizontal ratio
        bbox[3] /= self.__current_image_ratios[id_in_batch][0]  # shift to the bottom boundary divided by vertical ratio
        return bbox

    def translate_cat_id_to_coco(self, id: int, switch_to_indexing_from_1=True):
        """
        A function allowing for an easy translation of some networks' COCO category output that is in range of [0, 79]
        or [1, 80] (corresponding with number of 80 active COCO object categories in 2014 & 2017 sets). This translation
        is needed as PyCOCO evaluation tooling expects the original indexing [1, 90] with those unpredictable gaps...
        :param id: int, index representing category of COCO recognized object
        :param switch_to_indexing_from_1: bool, whether to switch to indexing beginning with 1 (switch it off if done
        already)
        :return: int, index as expected by COCO
        """
        coco_ids_map = {
            1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 13, 13: 14, 14: 15, 15: 16,
            16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 27, 26: 28, 27: 31, 28: 32,
            29: 33, 30: 34, 31: 35, 32: 36, 33: 37, 34: 38, 35: 39, 36: 40, 37: 41, 38: 42, 39: 43, 40: 44, 41: 46,
            42: 47, 43: 48, 44: 49, 45: 50, 46: 51, 47: 52, 48: 53, 49: 54, 50: 55, 51: 56, 52: 57, 53: 58, 54: 59,
            55: 60, 56: 61, 57: 62, 58: 63, 59: 64, 60: 65, 61: 67, 62: 70, 63: 72, 64: 73, 65: 74, 66: 75, 67: 76,
            68: 77, 69: 78, 70: 79, 71: 80, 72: 81, 73: 82, 74: 84, 75: 85, 76: 86, 77: 87, 78: 88, 79: 89, 80: 90
        }
        if switch_to_indexing_from_1:
            id += 1
        return coco_ids_map[id]

    def submit_bbox_prediction(self, id_in_batch, bbox, score, category):
        """
        A function meant for submitting a single bbox prediction for a given image.
        :param id_in_batch: int, id of an image in the currently processed batch that the provided bbox relates to
        :param bbox: list, list containing bbox coordinates
        :param score: float, value of the confidence in the prediction
        :param category: int, index of class / category in COCO order (starting with idx = 1)
        :return:
        """
        assert self._task == "bbox"
        instance = [self._current_image_ids[id_in_batch]] + self.rescale_bbox(id_in_batch, bbox) + [score, category]
        self._predictions.append(instance)

    def submit_mask_prediction(self, id_in_batch, bbox, score, category, mask):
        assert self._task == "segm"
        self._predictions.append({
            "image_id": self._current_image_ids[id_in_batch],
            "category_id": category,
            "bbox": self.rescale_bbox(id_in_batch, bbox),
            "score": score,
            "segmentation": mask
        })

    def summarize_accuracy(self):
        """
        A function summarizing the accuracy achieved on the images obtained with get_input_array() calls on which
        predictions done where supplied with submit_bbox_prediction() function.
        """
        if self._task == "bbox":
            predictions = np.array(self._predictions)
        elif self._task == "segm":
            predictions = self._predictions
        else:
            assert False
        predictions = self.__ground_truth.loadRes(predictions)
        coco_eval = COCOeval(self.__ground_truth, predictions, self._task)
        coco_eval.params.imgIds = self._image_ids[0:self._current_img]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        #print(f"\nAccuracy figures above calculated on the basis of {self._current_img} images.")
        return {"coco_map": coco_eval.stats[0]}


class COCODataset(COCOBaseDataset):
    """
    A class providing facilities to measure accuracy of models trained on COCO dataset.
    """

    def __init__(self,
                 batch_size: int, color_model: str, images_filename_base: str,
                 images_path=None, annotations_path=None, pre_processing=None, sort_ascending=False, order="NHWC",
                 task: str = "bbox"):
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
            try:
                images_path = os.environ["COCO_IMG_PATH"]
            except KeyError:
                download_coco_dataset()
                images_path = os.environ["COCO_IMG_PATH"]
        if annotations_path is None:
            try:
                annotations_path = os.environ["COCO_ANNO_PATH"]
            except KeyError:
                download_coco_labels()
                annotations_path = os.environ["COCO_ANNO_PATH"]

        self.__images_filename_base = images_filename_base
        self.__images_filename_ext = ".jpg"
        self.__images_path = images_path
        super().__init__(batch_size, color_model, annotations_path, pre_processing, order, sort_ascending, task)

    def _get_path_to_img(self):
        """
        A function providing path to the COCO image.
        :return: pathlib.PurePath object containing path to the image
        """
        try:
            image_id = self._image_ids[self._current_img]
        except IndexError:
            if os.environ.get("IGNORE_DATASET_LIMITS") == "1":
                if self.reset():
                    return self._get_path_to_img()
            raise utils.OutOfInstances("No more COCO images to process in the directory provided")
        self._current_image_ids.append(image_id)
        image_path = self.__images_filename_base[:-len(str(image_id))] + str(image_id) + self.__images_filename_ext
        self._current_img += 1
        return pathlib.PurePath(self.__images_path, image_path)
