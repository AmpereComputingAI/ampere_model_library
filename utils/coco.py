import os
import cv2
import sys
import json
import copy
import random
import decimal
import pathlib
import hashlib
import numpy as np
from cache.utils import get_cache_dir
import utils.miscellaneous as utils


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
    union_area = area_of_bbox_0 + area_of_bbox_1 - intersection_area
    assert union_area != 0.0, "area of union cannot be equal to 0"
    return intersection_area / union_area


class COCODataset:
    def __init__(self,
                 batch_size,
                 allow_distortion=True,
                 processed_annotations_path=None,
                 images_filename_base="000000000000",
                 pre_processing_func=None,
                 input_data_type="uint8",
                 run_in_loop=True):

        # TODO: hide some params from outside world
        self.batch_size = batch_size
        self.shape = None
        self.allow_distortion = allow_distortion
        self.pre_processing_func = pre_processing_func
        self.input_data_type = input_data_type
        self.run_in_loop = run_in_loop
        self.image_id = 0
        self.max_image_id = 1000000
        self.images_filename_base = images_filename_base
        self.images_filename_extension = ".jpg"
        self.ongoing_examples = self.__ExamplesTracker()
        # self.coco_mAP_lower_iou_thld = coco_mAP_lower_iou_thld
        # self.coco_mAP_upper_iou_thld = coco_mAP_upper_iou_thld
        self.coco_mAP_iou_thresholds = np.linspace(
            .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True).tolist()
        self.coco_recall_thresholds = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
        try:
            self.coco_directory = os.environ["COCO_DIR"]
        except KeyError:
            utils.print_goodbye_message_and_die("COCO dataset directory has not been specified with COCO_DIR flag")
        try:
            if processed_annotations_path is None:
                self.coco_annotations = self.__initialize_coco_annotations(
                    pathlib.PurePath(os.environ["COCO_ANNO_PATH"]))
            else:
                self.coco_annotations = self.__initialize_coco_annotations(
                    processed_annotations_path, already_processed=True)
            self.__num_categories = self.coco_annotations["max_category_id"] + 1
        except KeyError:
            utils.print_goodbye_message_and_die("COCO annotations path has not been specified with COCO_ANNO_PATH flag")
        self.__true_positives_matrix, self.__false_positives_matrix, self.__scores_container, self.__cat_population = \
            self.__init_acc_tracking_arrays()
        self.example_generator = self.__get_next_example()

    # def __get_num_of_iou_jumps(self):
    #    offset = 1  # between eg. 0.5 - 0.95 you have 10 variants when jumps equal 0.05, not 9
    #    return int(((self.coco_mAP_upper_iou_thld - self.coco_mAP_lower_iou_thld) / self.coco_mAP_iou_jump) + offset)

    def __init_list_of_lists(self, dim_0, dim_1):
        container = list()
        for _ in range(dim_0):
            cat_container = list()
            for _ in range(dim_1):
                cat_container.append(list())
            container.append(cat_container)
        return container

    def __init_acc_tracking_arrays(self):
        # although COCO indexing starts with 1 and some in-between indices
        # are missing in 2017 set (like categories under ids: 12, 26, ..., 91)
        # we don't want to bother translating them etc. so we just create an
        # array able to accommodate them under their native indices despite
        # it being bigger than necessary
        # categories_dim_size = self.coco_annotations["max_category_id"] + 1
        true_positives_container = self.__init_list_of_lists(len(self.coco_mAP_iou_thresholds), self.__num_categories)
        false_positives_container = self.__init_list_of_lists(len(self.coco_mAP_iou_thresholds), self.__num_categories)
        scores_container = self.__init_list_of_lists(len(self.coco_mAP_iou_thresholds), self.__num_categories)
        return true_positives_container, false_positives_container, scores_container, np.zeros(self.__num_categories)
        #return np.zeros(shape), np.full(shape, np.spacing(1)), np.zeros(shape)

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

        def submit_prediction(
                self, id_in_batch: int, score: float, bbox_in_coco_format: list, category_id: int, batch_size: int):
            self.predictions[len(self.predictions) - batch_size + id_in_batch].append(
                {"score": score, "bbox": bbox_in_coco_format, "category_id": category_id})

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

    def __initialize_coco_annotations(self, annotations_path, already_processed=False):
        if already_processed:
            with open(annotations_path) as annotations:
                return json.load(annotations)
        cached_annotations_path = pathlib.PurePath(
                get_cache_dir(), f"annotations_{get_hash_of_a_file(annotations_path)}.json")
        if pathlib.Path(cached_annotations_path).is_file():
            with open(cached_annotations_path) as annotations:
                return json.load(annotations)
        else:
            extracted_data = {
                "instances": dict(),
                "categories": dict(),
                # "num_categories": 0,
                "max_category_id": 0
            }
            extracted_instances = extracted_data["instances"]
            extracted_categories = extracted_data["categories"]
            with open(annotations_path) as annotations:
                annotations = json.load(annotations)
                for annotation in annotations["annotations"]:
                    ignore = 0
                    if "iscrowd" in annotation and annotation["iscrowd"] == 1:
                        ignore = 1
                    if "ignore" in annotation and annotation["ignore"] == 1:
                        ignore = 1
                    dict_with_bbox_data = {
                        "ignore": ignore,
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
                    # extracted_data["num_categories"] += 1
                    if category["id"] > extracted_data["max_category_id"]:
                        extracted_data["max_category_id"] = category["id"]
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
                utils.print_goodbye_message_and_die(
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

    def submit_bbox_prediction(self, id_in_batch: int, score: float, bbox_in_coco_format: list, category_id: int):
        """
        :param id_in_batch:
        :param score: float
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
        self.ongoing_examples.submit_prediction(id_in_batch, score, bbox_in_coco_format, category_id, self.batch_size)

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

    def __is_valid_instance(self, bbox):
        # instance is not valid if bbox has an area of 0
        # (as a result of cropping the image and moving the annotated instance outside)
        left, top, right, bottom = unpack_bbox_into_vars(bbox)
        if left == top == right == bottom:
            return False
        return True

    class MatchRegistry:
        # registry where reference bboxes are owners of matching predicted bboxes
        def __init__(self, pred_bboxes_num, pred_cat_ids, max_cat_id):
            self.__registry = list()
            self.__pred_ownership = self.PredOwnership(pred_bboxes_num)
            self.__pred_cat_ids = pred_cat_ids
            self.__max_cat_id_possible = max_cat_id

        class PredOwnership:
            def __init__(self, pred_bboxes_num):
                self.pred_ownership_array = np.full(pred_bboxes_num, -1)
                self.pred_num = pred_bboxes_num

            def get_owner(self, pred_id):
                return self.pred_ownership_array[pred_id]

            def register_owner(self, pred_id, owner_id):
                self.pred_ownership_array[pred_id] = owner_id

            def pred_has_owner(self, pred_id):
                if self.pred_ownership_array[pred_id] != -1:
                    return True
                return False

        def register(self, ref_bbox_item):
            self.__registry.append(ref_bbox_item)

        def __calc_status(self, registry, pred_ownership):
            num_matches = 0
            total_IoU = 0.0
            for i in range(pred_ownership.pred_num):
                if pred_ownership.pred_has_owner(i):
                    num_matches += 1
                    total_IoU += registry[pred_ownership.get_owner(i)].get_best_pred_IoU()
            return num_matches, total_IoU

        def optimize_matches(self, registry, pred_ownership):
            for i, ref_bbox in enumerate(registry):
                if ref_bbox.has_matches():
                    if pred_ownership.pred_has_owner(ref_bbox.get_best_pred_id()):
                        owner_id = pred_ownership.get_owner(ref_bbox.get_best_pred_id())
                        owner = registry[owner_id]
                        # if any of concurring reference bboxes still has
                        # alternative matches we explore the possibilities
                        if owner.matches_available() or ref_bbox.matches_available():
                            # variant 0 - we make ref_bbox surrender to current owner
                            # by removing pred from its matches and making recursive search
                            if registry[i].matches_available():
                                registry_0 = copy.deepcopy(registry)
                                pred_ownership_0 = copy.deepcopy(pred_ownership)
                                registry_0[i].move_to_next_match()
                                num_matches_0, total_IoU_0, registry_0, pred_ownership_0 = self.optimize_matches(
                                    registry_0, pred_ownership_0)
                            else:
                                num_matches_0, total_IoU_0 = self.__calc_status(registry, pred_ownership)
                                registry_0, pred_ownership_0 = registry, pred_ownership

                            # variant 1 - we make current owner surrender to ref bbox
                            if registry[owner_id].matches_available():
                                registry_1 = copy.deepcopy(registry)
                                pred_ownership_1 = copy.deepcopy(pred_ownership)
                                pred_ownership_1.register_owner(ref_bbox.get_best_pred_id(), ref_bbox.id)
                                registry_1[owner_id].move_to_next_match()
                                num_matches_1, total_IoU_1, registry_1, pred_ownership_1 = self.optimize_matches(
                                    registry_1, pred_ownership_1)
                            else:
                                pred_ownership_1 = copy.deepcopy(pred_ownership)
                                pred_ownership_1.register_owner(ref_bbox.get_best_pred_id(), ref_bbox.id)
                                num_matches_1, total_IoU_1 = self.__calc_status(registry, pred_ownership_1)
                                registry_1 = registry

                            if num_matches_0 < num_matches_1:
                                registry = registry_1
                                pred_ownership = pred_ownership_1
                            elif num_matches_0 > num_matches_1:
                                registry = registry_0
                                pred_ownership = pred_ownership_0
                            else:
                                if total_IoU_0 < total_IoU_1:
                                    registry = registry_1
                                    pred_ownership = pred_ownership_1
                                else:
                                    registry = registry_0
                                    pred_ownership = pred_ownership_0

                        # if both dont have any more alternatives we have to choose - we select the one with higher IoU
                        elif owner.get_best_pred_IoU() < ref_bbox.get_best_pred_IoU():
                            pred_ownership.register_owner(ref_bbox.get_best_pred_id(), ref_bbox.id)
                    else:
                        pred_ownership.register_owner(ref_bbox.get_best_pred_id(), ref_bbox.id)

            num_matches, total_IoU = self.__calc_status(registry, pred_ownership)
            return num_matches, total_IoU, registry, pred_ownership

        def summarize(self, IoU_index, tp_container, fp_container, scores_container, pred_scores, ignore_mask):
            _, _, self.__registry, self.__pred_ownership = self.optimize_matches(self.__registry, self.__pred_ownership)

            for i in range(self.__pred_ownership.pred_num):
                if ignore_mask[self.__pred_ownership.get_owner(i)] == 1:
                    continue
                scores_container[IoU_index][self.__pred_cat_ids[i]].append(pred_scores[i])
                if self.__pred_ownership.get_owner(i) == -1:
                    # pred does not have a match at given IoU threshold - false positive
                    tp_container[IoU_index][self.__pred_cat_ids[i]].append(0)
                    fp_container[IoU_index][self.__pred_cat_ids[i]].append(1)
                else:
                    tp_container[IoU_index][self.__pred_cat_ids[i]].append(1)
                    fp_container[IoU_index][self.__pred_cat_ids[i]].append(0)

    class UnmatchedRefBBox:
        def __init__(self, ref_bbox_id, category_id):
            self.id = ref_bbox_id
            self.category_id = category_id

        def has_matches(self):
            return False

    class RefBBox:
        def __init__(self, ref_bbox_id, category_id, array_with_IoUs, IoU_threshold):
            self.id = ref_bbox_id
            self.category_id = category_id
            # sort IoUs in descending order
            self.array_with_IoUs = -np.sort(-array_with_IoUs)
            # create an array with indices in original IoUs matrix for values in array_with_IoUs
            self.array_with_ids = np.argsort(-array_with_IoUs)
            # remove IoUs below the threshold
            ids_to_delete = np.argwhere(self.array_with_IoUs < IoU_threshold).flatten()
            self.array_with_IoUs = np.delete(self.array_with_IoUs, ids_to_delete)
            self.array_with_ids = np.delete(self.array_with_ids, ids_to_delete)
            self.current_pos = 0

        def has_matches(self):
            return True

        def get_best_pred_id(self):
            return self.array_with_ids[self.current_pos]

        def get_best_pred_IoU(self):
            return self.array_with_IoUs[self.current_pos]

        def move_to_next_match(self):
            self.current_pos += 1

        def matches_available(self):
            if self.current_pos + 1 >= self.array_with_IoUs.shape[0]:
                return False
            return True

    def __calculate_IoU_matrix(self, truth, pred):
        # this will contain every iou combination between genuine bboxes (dim 0) and predicted (dim 1)
        # iou_matrix = np.full((len(truth), len(pred)), float(-1))
        iou_matrix = np.zeros((len(truth), len(pred)))
        pred_cat_ids = np.full(len(pred), -1)
        pred_scores = np.full(len(pred), 0.)
        for true_instance in truth:
            for prediction in pred:
                pred_cat_ids[pred.index(prediction)] = prediction["category_id"]
                pred_scores[pred.index(prediction)] = prediction["score"]
                if true_instance["category_id"] != prediction["category_id"]:
                    continue
                iou_matrix[truth.index(true_instance)][pred.index(prediction)] = calc_iou(
                    true_instance["bbox"], prediction["bbox"])
        return iou_matrix, pred_cat_ids, pred_scores

    def __remove_nonvalid_instances(self, truth):
        new_truth = list()
        for true_instance in truth:
            if self.__is_valid_instance(true_instance["bbox"]):
                new_truth.append(true_instance)
        return new_truth

    def __calc_acc(self, ignore_mask, ref_cat_ids, pred_cat_ids, pred_scores, num_preds, IoU_matrix, IoU_threshold):
        match_registry = self.MatchRegistry(num_preds, pred_cat_ids, self.coco_annotations["max_category_id"])

        for i in range(len(ref_cat_ids)):
            array_with_IoUs = IoU_matrix[i]
            if array_with_IoUs.size > 0:
                if np.max(array_with_IoUs) >= IoU_threshold:
                    match_registry.register(
                        self.RefBBox(i, ref_cat_ids[i], array_with_IoUs, IoU_threshold))
                    continue
            match_registry.register(self.UnmatchedRefBBox(i, ref_cat_ids[i]))

        match_registry.summarize(
            self.coco_mAP_iou_thresholds.index(IoU_threshold),
            self.__true_positives_matrix,
            self.__false_positives_matrix,
            self.__scores_container,
            pred_scores,
            ignore_mask
        )

    def __coco_calculate_accuracy(self):
        for i in range(len(self.ongoing_examples.ground_truth)):
            truth = self.__remove_nonvalid_instances(self.ongoing_examples.ground_truth[i])
            ref_cat_ids = list()
            ignore_mask = list()
            for ref_bbox in truth:
                self.__cat_population[ref_bbox["category_id"]] += 1
                ref_cat_ids.append(ref_bbox["category_id"])
                ignore_mask.append(ref_bbox["ignore"])
            IoU_matrix, pred_cat_ids, pred_scores = self.__calculate_IoU_matrix(
                truth, self.ongoing_examples.predictions[i])
            for iou_thld in self.coco_mAP_iou_thresholds:
                self.__calc_acc(
                    ignore_mask, ref_cat_ids, pred_cat_ids, pred_scores, len(self.ongoing_examples.predictions[i]), IoU_matrix, iou_thld)

    def get_input_array(self, input_shape):
        self.shape = input_shape
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
        # we have to process our last request to get_input_tensor() before proceeding to summary
        self.__coco_calculate_accuracy()

        # let's go
        precision_matrix = np.full(
            (len(self.coco_mAP_iou_thresholds), self.__num_categories, len(self.coco_recall_thresholds)), np.nan)
        recall_matrix = np.full((len(self.coco_mAP_iou_thresholds), self.__num_categories), np.nan)
        #print(recall_matrix.shape)
        for i in range(len(self.coco_mAP_iou_thresholds)):
            for c in range(self.__num_categories):
                if self.__cat_population[c] == 0:
                    #if len(self.__false_positives_matrix[i][c]) > 0:
                    #    precision_matrix[i][c][:] = np.zeros((len(self.coco_recall_thresholds)))
                    continue

                recall_to_precision_array = np.zeros((len(self.coco_recall_thresholds)))
                if len(self.__true_positives_matrix[i][c]) > 0 or len(self.__false_positives_matrix[i][c]) > 0:
                    inds = np.argsort(-np.array(self.__scores_container[i][c]), kind='mergesort')
                    true_positives = np.cumsum(np.array(self.__true_positives_matrix[i][c])[inds])
                    false_positives = np.cumsum(np.array(self.__false_positives_matrix[i][c])[inds])
                    precision_levels = true_positives / (true_positives + false_positives + np.spacing(1))
                    #if i == 9 and c == 1:
                    #    print(true_positives)
                    #    print(false_positives)
                    #    print(precision_levels)
                    #    print("FFF")
                    for pos in reversed(range(1, len(precision_levels))):
                        if precision_levels[pos] > precision_levels[pos-1]:
                            precision_levels[pos-1] = precision_levels[pos]
                    recall_levels = true_positives / self.__cat_population[c]
                    # indices to input precisions under in 101-dimensional recall space
                    indices = np.searchsorted(recall_levels, self.coco_recall_thresholds, side="left")
                    for recall_id, idx in enumerate(indices):
                        try:
                            recall_to_precision_array[recall_id] = precision_levels[idx]
                        except IndexError:
                            # in case we exceed precision levels size by idx we stop
                            break
                    precision_matrix[i][c][:] = recall_to_precision_array
                    recall_matrix[i][c] = recall_levels[-1]
                else:
                    # no true positives but we know that cat_population (ie. ref bbox population) is bigger than 0
                    precision_matrix[i][c][:] = recall_to_precision_array
                    recall_matrix[i][c] = 0
        #for i in precision_matrix:
        #    print("HARAK")
        #    z = 0
        #    for c in i:
        #        if z == 1:
        #            print(z)
        #            print(c)
        #        z += 1
        print(np.nanmean(precision_matrix))
        print(np.nanmean(recall_matrix))
