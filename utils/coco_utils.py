import os
import cv2
import sys
import pathlib
import numpy as np


def print_goodbye_message_and_die(message):
    print(f"FAIL: {message}")
    sys.exit(1)


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
        try:
            self.coco_directory = os.environ["COCO_DIR"]
        except KeyError:
            print_goodbye_message_and_die("COCO dataset directory has not been specified with COCO_DIR flag")

    def __generate_coco_filename(self):
        image_id_as_str = str(self.image_id)
        return self.images_filename_base[:-len(image_id_as_str)] + image_id_as_str + self.images_filename_extension

    def __find_next_images_path(self):
        while self.image_id < self.max_image_id:
            potential_path = pathlib.PurePath(self.coco_directory, self.__generate_coco_filename())
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

        # applying scale factor if image is either smaller or larger by both dimensions
        if scale_factor:
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
        padded_array[lower_boundary_h:upper_boundary_h, lower_boundary_w:upper_boundary_w] = image_array
        return padded_array

    def __get_next_image(self):
        image_array = cv2.imread(self.__get_path_to_image())
        if self.allow_distortion:
            image_array = cv2.resize(image_array, self.shape)
        else:
            image_array = self.__resize_and_crop_image(image_array)
        self.image_id += 1
        return np.expand_dims(image_array, axis=0).astype(self.input_data_type)

    def get_input_array(self):
        input_array = self.__get_next_image()
        for _ in range(1, self.batch_size):
            input_array = np.concatenate((input_array, self.__get_next_image()), axis=0)
        if self.pre_processing_func:
            input_array = self.pre_processing_func(input_array)
        return input_array
