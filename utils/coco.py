import os
import cv2
import pathlib
import numpy as np
import utils.miscellaneous as utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


class ImageDataset:
    def __init__(self, allow_distortion=True):
        self.__allow_distortion = allow_distortion

    def __resize_and_crop_image(self, image_array, target_shape):
        image_height = image_array.shape[0]
        image_width = image_array.shape[1]
        target_height = target_shape[0]
        target_width = target_shape[1]

        if target_height < image_height and target_width < image_width:
            if image_height < image_width:
                scale_factor = target_height / image_height
            else:
                scale_factor = target_width / image_width
        elif target_height > image_height and target_width > image_width:
            if image_height < image_width:
                scale_factor = target_width / image_width
            else:
                scale_factor = target_height / image_height
        else:
            scale_factor = None

        if scale_factor:
            # applying scale factor if image is either smaller or larger by both dimensions
            image_array = cv2.resize(image_array,
                                     (int(image_width * scale_factor + 0.9999999999),
                                      int(image_height * scale_factor + 0.9999999999)))

        # padding - interpretable as black bars
        padded_array = np.zeros((*target_shape, 3))
        image_array = image_array[:target_height, :target_width]
        lower_boundary_h = int((target_height - image_array.shape[0]) / 2)
        upper_boundary_h = image_array.shape[0] + lower_boundary_h
        lower_boundary_w = int((target_width - image_array.shape[1]) / 2)
        upper_boundary_w = image_array.shape[1] + lower_boundary_w
        padded_array[lower_boundary_h:upper_boundary_h, lower_boundary_w:upper_boundary_w] = image_array
        return padded_array

    def __resize_image(self, image_array, target_shape):
        vertical_ratio = target_shape[0] / image_array.shape[0]
        horizontal_ratio = target_shape[1] / image_array.shape[1]
        return cv2.resize(image_array, target_shape), (vertical_ratio, horizontal_ratio)

    def __rescale_image(self, image_array, target_shape):
        if self.__allow_distortion:
            return self.__resize_image(image_array, target_shape)
        return self.__resize_and_crop_image(image_array, target_shape), None

    def __load_image(self, image_path, target_shape, get_resize_ratios=False):
        """
        target_shape = (height, width)

        :param image_path:
        :param target_shape:
        :return:
        """
        image_array = cv2.imread(str(image_path))
        assert image_array is not None
        image_array, resize_ratios = self.__rescale_image(image_array, target_shape)
        if get_resize_ratios:
            return np.expand_dims(image_array, axis=0), resize_ratios
        else:
            return np.expand_dims(image_array, axis=0)


class COCODataset(ImageDataset):
    def __init__(self,
                 batch_size, images_filename_base,
                 images_path=None, annotations_path=None,
                 allow_distortion=True, pre_processing_func=None, sort_ascending=False):

        if images_path is None:
            env_var = "COCO_IMG_PATH"
            images_path = utils.get_env_variable(
                env_var, f"Path to COCO images directory has not been specified with {env_var} flag")
        if annotations_path is None:
            env_var = "COCO_ANNO_PATH"
            annotations_path = utils.get_env_variable(
                env_var, f"Path to COCO annotations has not been specified with {env_var} flag")

        self.__batch_size = batch_size
        self.__images_path = images_path
        self.__annotations_path = annotations_path
        self.images_filename_base = images_filename_base
        self.images_filename_extension = ".jpg"
        self.__pre_processing_func = pre_processing_func
        self.__batch_size = batch_size
        self.__ground_truth = COCO(annotations_path)
        self.__current_img = 0
        self.__detections = list()
        self.__current_image_ids = list()
        self.__current_image_ratios = list()
        self.__image_ids = self.__ground_truth.getImgIds()
        if sort_ascending:
            self.__image_ids = sorted(self.__image_ids)
        self.available_images = len(self.__image_ids)
        assert allow_distortion, "Not implemented yet for COCO"
        super().__init__(allow_distortion)

    class OutOfCOCOImages(Exception):
        pass

    def __get_path_to_img(self):
        try:
            image_id = self.__image_ids[self.__current_img]
            #print(image_id)
        except IndexError:
            raise self.OutOfCOCOImages("No more COCO images to process in the directory provided")
        self.__current_image_ids.append(image_id)
        image_path = self.images_filename_base[:-len(str(image_id))] + str(image_id) + self.images_filename_extension
        self.__current_img += 1
        return pathlib.PurePath(self.__images_path, image_path)

    def __rescale_bbox(self, id_in_batch: int, bbox: list):
        bbox[0] /= self.__current_image_ratios[id_in_batch][1]  # left boundary divided by horizontal ratio
        bbox[1] /= self.__current_image_ratios[id_in_batch][0]  # top boundary divided by vertical ratio
        bbox[2] /= self.__current_image_ratios[id_in_batch][1]  # shift to the right divided by horizontal ratio
        bbox[3] /= self.__current_image_ratios[id_in_batch][0]  # shift to the bottom boundary divided by vertical ratio
        return bbox

    def __reset_containers(self):
        self.__current_image_ids = list()
        self.__current_image_ratios = list()

    def __load_image_and_store_ratios(self, target_shape):
        input_array, resize_ratios = self._ImageDataset__load_image(
            self.__get_path_to_img(), target_shape, get_resize_ratios=True)
        self.__current_image_ratios.append(resize_ratios)
        return input_array

    def get_input_array(self, target_shape):
        self.__reset_containers()
        input_array = self.__load_image_and_store_ratios(target_shape)
        for _ in range(1, self.__batch_size):
            input_array = np.concatenate((input_array, self.__load_image_and_store_ratios(target_shape)), axis=0)
        if self.__pre_processing_func:
            input_array = self.__pre_processing_func(input_array)
        return input_array

    def convert_bbox_to_coco_order(self, bbox, left=0, top=1, right=2, bottom=3, absolute=True):
        # sometimes networks return order of bbox boundary values in different order
        # than the default COCO left -> top -> right -> bottom
        left = bbox[left]
        top = bbox[top]
        right = bbox[right]
        bottom = bbox[bottom]
        if absolute:
            right -= left
            bottom -= top
        return [left, top, right, bottom]

    def submit_bbox_prediction(self, id_in_batch, bbox, score, category):
        instance = list()
        instance.append(self.__current_image_ids[id_in_batch])
        #print(bbox)
        #print(self.__rescale_bbox(id_in_batch, bbox))
        instance += self.__rescale_bbox(id_in_batch, bbox)
        instance.append(score)
        instance.append(category)
        self.__detections.append(instance)

    def summarize_accuracy(self):
        #print(np.array(self.__detections))
        detections = self.__ground_truth.loadRes(np.array(self.__detections))
        coco_eval = COCOeval(self.__ground_truth, detections, "bbox")
        coco_eval.params.imgIds = self.__image_ids[0:self.__current_img]
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        print(f"\nAccuracy figures above calculated on the basis of {self.__current_img} images.")



#test = COCODataset(batch_size=1)
#test.
