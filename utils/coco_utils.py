import os
import cv2
import sys
import json
import random
import decimal
import pathlib
import hashlib
import numpy as np
from cache.utils import get_cache_dir


def print_goodbye_message_and_die(message):
    print(f"FAIL: {message}")
    sys.exit(1)


def print_warning_message(message):
    print(f"CAUTION: {message}")


def get_hash_of_a_file(path_to_file):
    hash_md5 = hashlib.md5()
    with open(path_to_file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def calc_precision(true_positives, false_positives):
    tp_fp = true_positives + false_positives
    if tp_fp == 0:
        return None
    return true_positives / tp_fp


def calc_recall(true_positives, false_negatives):
    tp_fn = true_positives + false_negatives
    if tp_fn == 0:
        return None
    return true_positives / tp_fn


def unpack_bbox_into_vars(bbox: list):
    return bbox[0], bbox[1], bbox[2], bbox[3]


def calc_overlapping_part_of_two_line_segments(a_0, b_0, a_1, b_1):
    # a denotes smaller of values a and b marking end points of given (_i) line segment
    assert a_0 <= b_0 and a_1 <= b_1, "a value cannot be bigger than b"
    return min(b_0, b_1) - max(a_0, a_1)


def frange(start_value, end_value, jump):
    while start_value < end_value + jump:
        yield float(start_value)
        start_value += jump


def calc_iou(bbox_0: list, bbox_1: list):
    left_0, top_0, right_0, bottom_0 = unpack_bbox_into_vars(bbox_0)
    left_1, top_1, right_1, bottom_1 = unpack_bbox_into_vars(bbox_1)
    if right_0 <= left_1:
        return 0.0
    if right_1 <= left_0:
        return 0.0
    if bottom_0 <= top_1:
        return 0.0
    if bottom_1 <= top_0:
        return 0.0
    horizontal_common_length = calc_overlapping_part_of_two_line_segments(left_0, right_0, left_1, right_1)
    vertical_common_length = calc_overlapping_part_of_two_line_segments(top_0, bottom_0, top_1, bottom_1)
    intersection_area = horizontal_common_length * vertical_common_length
    area_of_bbox_0 = (right_0 - left_0) * (bottom_0 - top_0)
    area_of_bbox_1 = (right_1 - left_1) * (bottom_1 - top_1)
    union_area = area_of_bbox_0 + area_of_bbox_1 - 2 * intersection_area
    assert union_area != 0.0, "area of union cannot be equal to 0"
    return intersection_area / union_area


class COCODataset:
    def __init__(self,
                 batch_size, shape,
                 allow_distortion=True,
                 pre_processing_func=None,
                 input_data_type="uint8",
                 run_in_loop=True,
                 coco_mAP_lower_iou_thld=0.5,
                 coco_mAP_upper_iou_thld=0.95,
                 coco_mAP_iou_jump=0.05,
                 coco_recall_granularity=101):

        # TODO: hide some params from outside world
        self.batch_size = batch_size
        self.shape = shape
        self.allow_distortion = allow_distortion
        self.pre_processing_func = pre_processing_func
        self.input_data_type = input_data_type
        self.run_in_loop = run_in_loop
        self.image_id = 0
        self.max_image_id = 1000000
        self.images_filename_base = "000000000000"
        self.images_filename_extension = ".jpg"
        self.ongoing_examples = None
        # self.coco_mAP_lower_iou_thld = coco_mAP_lower_iou_thld
        # self.coco_mAP_upper_iou_thld = coco_mAP_upper_iou_thld
        self.coco_mAP_iou_thresholds = list(frange(coco_mAP_lower_iou_thld, coco_mAP_upper_iou_thld, coco_mAP_iou_jump))
        self.coco_recall_granularity = coco_recall_granularity
        try:
            self.coco_directory = os.environ["COCO_DIR"]
        except KeyError:
            print_goodbye_message_and_die("COCO dataset directory has not been specified with COCO_DIR flag")
        try:
            self.coco_annotations = self.__initialize_coco_annotations(pathlib.PurePath(os.environ["COCO_ANNO_PATH"]))
            self.accumulated_precision_matrix, self.occurrences_matrix = self.__init_acc_tracking_arrays()
            self.example_generator = self.__get_next_example()
        except KeyError:
            print_goodbye_message_and_die("COCO annotations path has not been specified with COCO_ANNO_PATH flag")

    # def __get_num_of_iou_jumps(self):
    #    offset = 1  # between eg. 0.5 - 0.95 you have 10 variants when jumps equal 0.05, not 9
    #    return int(((self.coco_mAP_upper_iou_thld - self.coco_mAP_lower_iou_thld) / self.coco_mAP_iou_jump) + offset)

    def __init_acc_tracking_arrays(self):
        # although COCO indexing starts with 1 and some in-between indices
        # are missing in 2017 set (like categories under ids: 12, 26, ..., 91)
        # we don't want to bother translating them etc. so we just create an
        # array able to accommodate them under their native indices despite
        # it being bigger than necessary
        #categories_dim_size = self.coco_annotations["max_category_id"] + 1
        shape = (self.coco_recall_granularity, len(self.coco_mAP_iou_thresholds))
        accumulated_precision_matrix = np.full(shape, np.nan)
        occurrences_matrix = np.zeros(shape)
        return accumulated_precision_matrix, occurrences_matrix

    class __ExamplesTracker:
        def __init__(self):
            self.ground_truth = dict()
            self.predictions = dict()
            self.example_id = -1

        def __convert_annotations(self):
            # eg. 'bbox': [473.07, 395.93, 38.65, 28.67]
            # value under idx 2/3 is shift to the right/bottom of value under idx 0/1
            # after conversion this changes to [473.07, 395.93, 38.65+473.07, 28.67+395.93]
            for instance in self.ground_truth[self.example_id]:
                instance["bbox"][2] += instance["bbox"][0]
                instance["bbox"][3] += instance["bbox"][1]

        def track_example(self, annotations):
            self.example_id += 1
            self.ground_truth[self.example_id] = annotations
            self.__convert_annotations()
            self.predictions[self.example_id] = list()

        def rescale_annotations(self, shape,
                                vertical_scale_factor, horizontal_scale_factor,
                                vertical_shift=0, horizontal_shift=0):
            for instance in self.ground_truth[self.example_id]:
                bbox = np.array(instance["bbox"]).astype("float32")
                if vertical_scale_factor and horizontal_scale_factor:
                    # if image was resized (w/ or w/o aspect ratio change)
                    # we need to appropriately correct bboxes
                    bbox *= np.array([horizontal_scale_factor, vertical_scale_factor,
                                      horizontal_scale_factor, vertical_scale_factor])
                if vertical_shift > 0 or horizontal_shift > 0:
                    # if black bars were applied we need to shift
                    bbox += np.array([horizontal_shift, vertical_shift,
                                      horizontal_shift, vertical_shift])
                # if both dimensions are equal we can use clip op to cap boxes at the edges of new image
                if shape[0] == shape[1]:
                    bbox = np.clip(bbox, 0, shape[0] - 1)
                # otherwise we have to go in alternating order (left boundary -> top -> right -> bottom)
                # and cap "by hand"
                else:
                    for i in range(4):
                        if i % 2 == 0:
                            bbox[i] = bbox[i] if bbox[i] < shape[1] else shape[1] - 1
                        else:
                            bbox[i] = bbox[i] if bbox[i] < shape[0] else shape[0] - 1
                instance["bbox"] = list(bbox)

    def convert_bbox_to_coco_order(self, bbox, left=0, top=1, right=2, bottom=3):
        # sometimes networks return order of bbox boundary values in different order
        # than the default COCO left -> top -> right -> bottom
        return [bbox[left], bbox[top], bbox[right], bbox[bottom]]

    def __initialize_coco_annotations(self, annotations_path):
        cached_annotations_path = pathlib.PurePath(
            get_cache_dir(), f"annotations_{get_hash_of_a_file(annotations_path)}.json")
        if pathlib.Path(cached_annotations_path).is_file():
            with open(cached_annotations_path) as annotations:
                return json.load(annotations)
        else:
            extracted_data = {
                "instances": dict(),
                "categories": dict(),
                #"num_categories": 0,
                #"max_category_id": 0
            }
            extracted_instances = extracted_data["instances"]
            extracted_categories = extracted_data["categories"]
            with open(annotations_path) as annotations:
                annotations = json.load(annotations)
                for annotation in annotations["annotations"]:
                    dict_with_bbox_data = {
                        "bbox": annotation["bbox"],
                        "category_id": annotation["category_id"]
                    }
                    # if the data related to the given image is already present append another bbox to the existing set
                    if annotation["image_id"] in extracted_instances:
                        extracted_instances[annotation["image_id"]].append(dict_with_bbox_data)
                    # else initialize the set for a given image with current bbox
                    else:
                        extracted_instances[annotation["image_id"]] = [dict_with_bbox_data]
                for category in annotations["categories"]:
                    #extracted_data["num_categories"] += 1
                    #if category["id"] > extracted_data["max_category_id"]:
                    #    extracted_data["max_category_id"] = category["id"]
                    extracted_categories[category["id"]] = category["name"]
            with open(cached_annotations_path, "w") as cached_annotations_file:
                json.dump(extracted_data, cached_annotations_file)
            return extracted_data

    def __generate_coco_filename(self, image_id: int):
        return self.images_filename_base[:-len(str(image_id))] + str(image_id) + self.images_filename_extension

    def __find_next_images_path(self):
        while self.image_id < self.max_image_id:
            potential_path = pathlib.PurePath(self.coco_directory, self.__generate_coco_filename(self.image_id))
            if pathlib.Path(potential_path).is_file():
                return potential_path
            self.image_id += 1
        return None

    def __get_path_to_image(self):
        path = self.__find_next_images_path()
        if not path:
            if self.run_in_loop:
                self.image_id = 0
                path = self.__find_next_images_path()
            if not path:
                print_goodbye_message_and_die(
                    f"Either end of dataset has been reached or "
                    f"files were not found under the directory {self.coco_directory}")
        return str(path)

    def __resize_and_crop_image(self, image_array):
        image_height = image_array.shape[0]
        image_width = image_array.shape[1]
        target_height = self.shape[0]
        target_width = self.shape[1]

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
        padded_array = np.zeros((self.shape[0], self.shape[1], 3))
        image_array = image_array[:target_height, :target_width]
        lower_boundary_h = int((target_height - image_array.shape[0]) / 2)
        upper_boundary_h = image_array.shape[0] + lower_boundary_h
        lower_boundary_w = int((target_width - image_array.shape[1]) / 2)
        upper_boundary_w = image_array.shape[1] + lower_boundary_w
        self.ongoing_examples.rescale_annotations(
            self.shape, scale_factor, scale_factor, lower_boundary_h, lower_boundary_w)
        padded_array[lower_boundary_h:upper_boundary_h, lower_boundary_w:upper_boundary_w] = image_array
        return padded_array

    def submit_bbox_prediction(self, id_in_batch: int, bbox_in_coco_format: list, category_id: int):
        """
        :param id_in_batch:
        :param bbox_in_coco_format:
        :param category_id:
        :return:
        """
        assert type(id_in_batch) is int, "id_in_batch should be provided as a single int value " \
                                         "representing position of image that submission is related " \
                                         "to in the processed batch"
        assert id_in_batch < self.batch_size, f"id_in_batch value exceeds range of batch declared " \
                                              f"- requested: {id_in_batch}; available range: 0-{self.batch_size - 1}"
        assert type(bbox_in_coco_format) is list, "bbox should be provided as a list " \
                                                  "eg. [463.66, 87.198135, 524.76, 225.71562] " \
                                                  "ie. [left_boundary, top_b, right_b, bottom_b]"
        assert len(bbox_in_coco_format) == 4, f"bbox should be provided as a list of 4 values " \
                                              f"- {len(bbox_in_coco_format)} provided"
        assert type(category_id) is int, "category_id should be provided as a single int value"
        self.ongoing_examples.predictions[id_in_batch].append(
            {"bbox": bbox_in_coco_format, "category_id": category_id})

    def __get_next_image(self):
        image_id, annotations = next(self.example_generator)
        self.ongoing_examples.track_example(annotations)
        image_array = cv2.imread(self.__get_path_to_image_under_id(image_id))
        assert image_array is not None, "image failed to load, possible causes: " \
                                        "a) supplied annotations don't match image set, " \
                                        "b) wrong directory to image set"
        image_array = self.__rescale_image(image_array)
        return np.expand_dims(image_array, axis=0).astype(self.input_data_type)

    def __rescale_image(self, image_array):
        if self.allow_distortion:
            self.ongoing_examples.rescale_annotations(
                self.shape, self.shape[0] / image_array.shape[0], self.shape[1] / image_array.shape[1])
            return cv2.resize(image_array, self.shape)
        return self.__resize_and_crop_image(image_array)

    def __instance_was_predicted(self, id_in_batch, instance, iou_threshold):
        iou = 0.0
        for pred in self.ongoing_examples.predictions[id_in_batch]:
            if pred["category_id"] == instance["category_id"]:
                iou = max(iou, calc_iou(instance["bbox"], pred["bbox"]))
        if iou > iou_threshold:
            return True
        return False

    def __valid_instance(self, bbox):
        # instance is not valid if bbox has an area of 0
        # (as a result of cropping the image and moving the annotated instance outside)
        left, top, right, bottom = unpack_bbox_into_vars(bbox)
        if left == top == right == bottom:
            return False
        return True

    def __push_to_accuracy_matrices(self, precision, recall, iou_threshold):
        recall_dim_idx = int(recall*100)
        iou_dim_idx = self.coco_mAP_iou_thresholds.index(iou_threshold)
        if np.isnan(self.accumulated_precision_matrix[recall_dim_idx][iou_dim_idx]):
            self.accumulated_precision_matrix[recall_dim_idx][iou_dim_idx] = precision
        else:
            self.accumulated_precision_matrix[recall_dim_idx][iou_dim_idx] += precision
        self.occurrences_matrix[recall_dim_idx][iou_dim_idx] += 1

    def __coco_calculate_prev_batch_accuracy(self):
        for iou_thld in self.coco_mAP_iou_thresholds:
            for i in range(self.batch_size):
                true_positives = 0
                false_negatives = 0
                instances = self.ongoing_examples.ground_truth[i]
                for instance in instances:
                    if not self.__valid_instance(instance["bbox"]):
                        continue
                    if self.__instance_was_predicted(i, instance, iou_thld):
                        true_positives += 1
                    else:
                        false_negatives += 1
                false_positives = len(self.ongoing_examples.predictions[i]) - true_positives
                precision = calc_precision(true_positives, false_positives)
                recall = calc_recall(true_positives, false_negatives)
                if not recall and not precision:
                    # image does not contain any instances and network didn't make a mistake to say otherwise
                    continue
                if recall and not precision:
                    # image does contain instances but network didn't see any
                    precision = 0.0
                if not recall and precision:
                    # image does not contain any instances but network said it does, arbitrary decision: skip
                    continue
                if precision > 1.0:
                    print("this requires fixing")
                else:
                    self.__push_to_accuracy_matrices(precision, recall, iou_thld)

    def get_input_array(self):
        if self.ongoing_examples:
            self.__coco_calculate_prev_batch_accuracy()
        self.ongoing_examples = self.__ExamplesTracker()
        input_array = self.__get_next_image()
        for _ in range(1, self.batch_size):
            input_array = np.concatenate((input_array, self.__get_next_image()), axis=0)
        if self.pre_processing_func:
            input_array = self.pre_processing_func(input_array)
        return input_array

    def __get_next_example(self):
        for image_id in self.coco_annotations["instances"]:
            yield image_id, self.coco_annotations["instances"][image_id]

    def __get_path_to_image_under_id(self, image_id):
        return str(pathlib.PurePath(self.coco_directory, self.__generate_coco_filename(image_id)))

    def summarize_accuracy(self):
        self.__coco_calculate_prev_batch_accuracy()
        for i in self.accumulated_precision_matrix:
            print(i)
        print("PROSZE STOP")
        avg_precision_matrix = np.divide(self.accumulated_precision_matrix, self.occurrences_matrix)
        for i in reversed(range(1, self.coco_recall_granularity)):
            # we go in reversed order applying interpolation along recall axis
            # - precision value will only increase with the decrease of recall
            avg_precision_matrix[i-1] = np.nanmax(avg_precision_matrix[i-1:i+1], axis=0)
        coco_mAP = np.nanmean(avg_precision_matrix)
        print(coco_mAP)
