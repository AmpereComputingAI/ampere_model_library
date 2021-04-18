import cv2
import pathlib
import numpy as np
import utils.misc as utils
from utils.dataset import ImageDataset
import utils.pre_processing as pp
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class COCODataset(ImageDataset):
    """
    A class providing facilities to measure accuracy of object detection models trained on COCO dataset.
    """
    def __init__(self,
                 batch_size: int, images_filename_base: str,
                 images_path=None, annotations_path=None, pre_processing_approach=None, sort_ascending=False):
        """
        A function initializing the class.

        :param batch_size: int, size of batch intended to be processed
        :param images_filename_base: str, default name of image files in given COCO dataset,
        eg. "COCO_val2014_000000000000"
        :param images_path: str, path to directory containing COCO images
        :param annotations_path: str, path to file containing COCO annotations
        :param pre_processing_func: python function, function handling pre-processing of an array containing image data
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

        self.__batch_size = batch_size
        self.__images_filename_base = images_filename_base
        self.__images_filename_ext = ".jpg"
        self.__images_path = images_path
        self.__annotations_path = annotations_path
        self.__pre_processing_approach = pre_processing_approach
        self.__ground_truth = COCO(annotations_path)
        self.__current_img = 0
        self.__detections = list()
        self.__current_image_ids = list()
        self.__current_image_ratios = list()
        self.__image_ids = self.__ground_truth.getImgIds()
        if sort_ascending:
            self.__image_ids = sorted(self.__image_ids)
        self.available_images_count = len(self.__image_ids)
        super().__init__()

    class OutOfCOCOImages(Exception):
        """
        An exception class being raised as an error in case of lack of further images to process by the pipeline.
        """
        pass

    def __get_path_to_img(self):
        """
        A function providing path to the COCO image.

        :return: pathlib.PurePath object containing path to the image
        """
        try:
            image_id = self.__image_ids[self.__current_img]
        except IndexError:
            raise self.OutOfCOCOImages("No more COCO images to process in the directory provided")
        self.__current_image_ids.append(image_id)
        image_path = self.__images_filename_base[:-len(str(image_id))] + str(image_id) + self.__images_filename_ext
        self.__current_img += 1
        return pathlib.PurePath(self.__images_path, image_path)

    def __rescale_bbox(self, id_in_batch: int, bbox: list):
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

    def __reset_containers(self):
        """
        A function resetting containers (lists) containing data on the on-going batch.
        """
        self.__current_image_ids = list()
        self.__current_image_ratios = list()

    def __load_image_and_store_ratios(self, target_shape):
        """
        A function loading single image and storing rescale ratios.

        :param target_shape: tuple of intended image shape (height, width)
        :return: numpy array containing rescaled image data
        """
        input_array, resize_ratios = self._ImageDataset__load_image(
            self.__get_path_to_img(), target_shape)
        self.__current_image_ratios.append(resize_ratios)
        return input_array

    def get_input_array(self, target_shape):
        """
        A function returning an array containing pre-processed rescaled image's or multiple images' data.

        :param target_shape: tuple of intended image shape (height, width)
        :return: numpy array containing rescaled, pre-processed image data of batch size requested at class
        initialization
        """
        self.__reset_containers()
        input_array = np.empty([self.__batch_size, *target_shape, 3])  # NHWC order
        for i in range(self.__batch_size):
            input_array[i] = self.__load_image_and_store_ratios(target_shape)
        if self.__pre_processing_approach:
            input_array = pp.preprocess(input_array, self.__pre_processing_approach)
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

    def submit_bbox_prediction(self, id_in_batch, bbox, score, category):
        """
        A function meant for submitting a single bbox prediction for a given image.

        :param id_in_batch: int, id of an image in the currently processed batch that the provided bbox relates to
        :param bbox: list, list containing bbox coordinates
        :param score: float, value of the confidence in the prediction
        :param category: int, index of class / category in COCO order (starting with idx = 1)
        :return:
        """
        instance = list()
        instance.append(self.__current_image_ids[id_in_batch])
        instance += self.__rescale_bbox(id_in_batch, bbox)
        instance.append(score)
        instance.append(category)
        self.__detections.append(instance)

    def summarize_accuracy(self):
        """
        A function summarizing the accuracy achieved on the images obtained with get_input_array() calls on which
        predictions done where supplied with submit_bbox_prediction() function.
        """
        detections = self.__ground_truth.loadRes(np.array(self.__detections))
        coco_eval = COCOeval(self.__ground_truth, detections, "bbox")
        coco_eval.params.imgIds = self.__image_ids[0:self.__current_img]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        print(f"\nAccuracy figures above calculated on the basis of {self.__current_img} images.")
