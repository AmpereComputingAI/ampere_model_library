import os
import cv2
import sys
import json
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


class COCODataset:
    def __init__(self,
                 batch_size, shape,
                 allow_distortion=True, pre_processing_func=None, input_data_type="uint8", run_in_loop=True):
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
        self.current_examples = None
        self.coco_annotations = None
        try:
            self.coco_directory = os.environ["COCO_DIR"]
        except KeyError:
            print_goodbye_message_and_die("COCO dataset directory has not been specified with COCO_DIR flag")
        try:
            self.__initialize_coco_annotations(pathlib.PurePath(os.environ["COCO_ANNO_PATH"]))
            self.example_generator = self.__get_next_example()
        except KeyError:
            print_goodbye_message_and_die("COCO annotations path has not been specified with COCO_ANNO_PATH flag")

    class __ExamplesTracker:
        def __init__(self):
            self.ground_truth = dict()
            self.predictions = dict()
            self.example_id = -1

        def __convert_annotations(self):
            # eg. 'bbox': [473.07, 395.93, 38.65, 28.67]
            # value under idx 2/3 is shift to the right/bottom of value under idx 0/1
            # after conversion this changes to [473.07, 395.93, 473.07+38.65, 395.93+28.67]
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
                    # if image was resized (w/ or w/o aspect ratio change) we need to appropriately correct bboxes
                    bbox *= np.array([horizontal_scale_factor, vertical_scale_factor,
                                      horizontal_scale_factor, vertical_scale_factor])
                if vertical_shift > 0 or horizontal_shift > 0:
                    # if black bars were applied we need to shift
                    bbox += np.array([horizontal_shift, vertical_shift,
                                      horizontal_shift, vertical_shift])
                # if both dimensions are equal we can use clip op to cap boxes at the edges of new image
                if shape[0] == shape[1]:
                    bbox = np.clip(bbox, 0, shape[0]-1)
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
        return [bbox[left], bbox[top], bbox[right], bbox[bottom]]

    def __initialize_coco_annotations(self, annotations_path):
        cached_annotations_path = pathlib.PurePath(
            get_cache_dir(), f"annotations_{get_hash_of_a_file(annotations_path)}.json")
        if pathlib.Path(cached_annotations_path).is_file():
            with open(cached_annotations_path) as annotations:
                self.coco_annotations = json.load(annotations)
        else:
            extracted_data = dict()
            with open(annotations_path) as annotations:
                annotations = json.load(annotations)
                for annotation in annotations["annotations"]:
                    dict_with_bbox_data = {
                        "bbox": annotation["bbox"],
                        "category_id": annotation["category_id"]
                    }
                    # if the data related to the given image is already present append another bbox to the existing set
                    if annotation["image_id"] in extracted_data:
                        extracted_data[annotation["image_id"]].append(dict_with_bbox_data)
                    # else initialize the set for a given image with current bbox
                    else:
                        extracted_data[annotation["image_id"]] = [dict_with_bbox_data]
            with open(cached_annotations_path, "w") as cached_annotations_file:
                json.dump(extracted_data, cached_annotations_file)
            self.__initialize_coco_annotations(annotations_path)

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
        self.current_examples.rescale_annotations(
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
        self.current_examples.predictions[id_in_batch].append({"bbox": bbox_in_coco_format, "category_id": category_id})

    def __get_next_image(self):
        image_id, annotations = next(self.example_generator)
        self.current_examples.track_example(annotations)
        image_array = cv2.imread(self.__get_path_to_image_under_id(image_id))
        image_array = self.__rescale_image(image_array)
        return np.expand_dims(image_array, axis=0).astype(self.input_data_type)

    def __rescale_image(self, image_array):
        if self.allow_distortion:
            self.current_examples.rescale_annotations(
                self.shape, self.shape[0] / image_array.shape[0], self.shape[1] / image_array.shape[1])
            image_array = cv2.resize(image_array, self.shape)
        else:
            image_array = self.__resize_and_crop_image(image_array)
        # cv2.imwrite("byt.jpg", image_array)
        return image_array

    def get_input_array(self):
        self.current_examples = self.__ExamplesTracker()
        input_array = self.__get_next_image()
        for _ in range(1, self.batch_size):
            input_array = np.concatenate((input_array, self.__get_next_image()), axis=0)
        if self.pre_processing_func:
            input_array = self.pre_processing_func(input_array)
        return input_array

    def __get_next_example(self):
        for image_id in self.coco_annotations:
            yield image_id, self.coco_annotations[image_id]

    def __get_path_to_image_under_id(self, image_id):
        return str(pathlib.PurePath(self.coco_directory, self.__generate_coco_filename(image_id)))

    def evaluate_accuracy(self, category_to_bbox_maps):
        return None
