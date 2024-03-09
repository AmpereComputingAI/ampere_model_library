# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
# Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2021 The MLPerf Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import json
import pathlib
import numpy as np
import nibabel as nib
from pathlib import Path
from scipy import signal
import utils.misc as utils
from utils.cv.kits_preprocessing import preprocess_with_multiproc, ROI_SHAPE, SLIDE_OVERLAP_FACTOR
from utils.helpers import Dataset


class KiTS19(Dataset):
    """
    A class providing facilities for preprocessing and postprocessing of KiTS19 dataset.
    """

    def __init__(self, dataset_dir_path=None):

        if dataset_dir_path is None:
            env_var = "KITS19_DATASET_PATH"
            dataset_dir_path = utils.get_env_variable(
                env_var, f"Path to KiTS19 dataset directory has not been specified with {env_var} flag")

        self.__dataset_dir_path = dataset_dir_path
        self.__preprocessed_files_dir_path = Path(self.__dataset_dir_path, "preprocessed")

        if not self.__preprocessed_files_dir_path.is_dir():
            self.__preprocess()

        cases_info = json.load(open(Path(self.__preprocessed_files_dir_path, "cases.json")))
        self.__case_ids = cases_info["cases"]
        self.__case_ids.sort()
        self.available_instances = cases_info["inferences_needed"]

        self.__current_img_id = 0
        self.__current_image = self.__Image()

        self.__kidney_score = 0.
        self.__tumor_score = 0.

    def __preprocess(self):
        """
        Function delegating the offline pre-processing work to slightly modified MLCommons script.
        """
        class ArgsSubstitute:
            def __init__(self, raw_data_dir, results_dir, num_proc=4):
                self.data_dir = raw_data_dir
                self.results_dir = results_dir
                self.mode = "preprocess"
                self.calibration = False
                self.num_proc = num_proc

        preprocess_with_multiproc(ArgsSubstitute(self.__dataset_dir_path, self.__dataset_dir_path))

    def __get_path_to_img(self):
        """
        A function providing path to the KiTS19 image.

        :return: pathlib.PurePath object containing path to the image
        """
        try:
            case_id = self.__case_ids[self.__current_img_id]
        except IndexError:
            raise utils.OutOfInstances("No more KiTS19 images to process in the directory provided")
        return pathlib.PurePath(self.__preprocessed_files_dir_path, case_id, "imaging.nii.gz")

    class __Image:
        """
        Class providing facilities for processing of currently considered image - such as splitting it into parts.
        """

        def __init__(self):
            self.all_issued = False
            self.empty = True
            self.__norm_patch = self.__gen_norm_patch()

        def __gen_norm_patch(self, std_factor=0.125):
            """
            Function calculating 3D gaussian kernel for normalization.
            """
            gaussian1d_0 = signal.gaussian(ROI_SHAPE[0], std_factor * ROI_SHAPE[0])
            gaussian1d_1 = signal.gaussian(ROI_SHAPE[1], std_factor * ROI_SHAPE[1])
            gaussian1d_2 = signal.gaussian(ROI_SHAPE[2], std_factor * ROI_SHAPE[2])
            gaussian2d = np.outer(gaussian1d_0, gaussian1d_1)
            gaussian3d = np.outer(gaussian2d, gaussian1d_2)
            gaussian3d = gaussian3d.reshape(*ROI_SHAPE)
            gaussian3d = np.cbrt(gaussian3d)
            gaussian3d /= gaussian3d.max()
            return gaussian3d

        def __populate_norm_map(self, norm_map_array):
            for i, j, k in self.__slice_indices:
                norm_map_slice = norm_map_array[
                                 ...,
                                 i:(ROI_SHAPE[0] + i),
                                 j:(ROI_SHAPE[1] + j),
                                 k:(ROI_SHAPE[2] + k)
                                 ]
                norm_map_slice += self.__norm_patch
            return norm_map_array

        def assign(self, image):
            self.__full_image = np.expand_dims(image, axis=0).astype("float32")
            self.all_issued = False
            self.empty = False
            self.__slice_indices = list()
            self.__current_slice_id = 0

            assert len(ROI_SHAPE) == 3 and any(ROI_SHAPE) and all(dim > 0 for dim in ROI_SHAPE), \
                f"Need proper ROI shape! The current ROI shape is: {ROI_SHAPE}"

            assert 0 < SLIDE_OVERLAP_FACTOR < 1, \
                f"Need sliding window overlap factor in (0,1)! The current overlap factor is: {SLIDE_OVERLAP_FACTOR}"

            image_shape = self.__full_image.shape[1:]
            dims = len(image_shape)
            strides = [int(ROI_SHAPE[i] * (1 - SLIDE_OVERLAP_FACTOR)) for i in range(dims)]
            size = [(image_shape[i] - ROI_SHAPE[i]) // strides[i] + 1 for i in range(dims)]

            for i in range(0, strides[0] * size[0], strides[0]):
                for j in range(0, strides[1] * size[1], strides[1]):
                    for k in range(0, strides[2] * size[2], strides[2]):
                        self.__slice_indices.append((i, j, k))

            self.__result = np.zeros(shape=(1, 3, *image_shape), dtype=self.__full_image.dtype)
            self.__norm_map = self.__populate_norm_map(np.zeros_like(self.__result))

        def get_next_input_slice(self):
            assert self.all_issued is False and self.empty is False
            i, j, k = self.__slice_indices[self.__current_slice_id]
            return self.__full_image[
                          ...,
                          i:(ROI_SHAPE[0] + i),
                          j:(ROI_SHAPE[1] + j),
                          k:(ROI_SHAPE[2] + k)
                          ]

        def accumulate_result_slice(self, output):
            i, j, k = self.__slice_indices[self.__current_slice_id]
            self.__result[
            ...,
            i:(ROI_SHAPE[0] + i),
            j:(ROI_SHAPE[1] + j),
            k:(ROI_SHAPE[2] + k)
            ] += output * self.__norm_patch
            self.__current_slice_id += 1
            if self.__current_slice_id == len(self.__slice_indices):
                self.all_issued = True

        def get_final_result(self):
            self.__result /= self.__norm_map
            return np.argmax(self.__result, axis=1).astype(np.uint8)

    def reset(self):
        self.__current_img_id = 0
        self.__current_image = self.__Image()
        self.__kidney_score = 0.
        self.__tumor_score = 0.
        return True

    def get_input_array(self):
        """
        A function returning an array containing slice of pre-processed image.
        """
        if self.__current_image.all_issued or self.__current_image.empty:
            self.__current_image.assign(nib.load(self.__get_path_to_img()).get_fdata())
        return self.__current_image.get_next_input_slice()

    def __get_gt_path(self):
        try:
            case_id = self.__case_ids[self.__current_img_id]
        except IndexError:
            raise utils.OutOfInstances("No more KiTS19 images to process in the directory provided")
        return pathlib.PurePath(self.__preprocessed_files_dir_path, case_id, "segmentation.nii.gz")

    def __calc_dice_score(self, prediction, target):
        """
        Calculates DICE score of prediction against target, for classes as many as case
        One-hot encoded form of case/prediction used for easier handling
        Background case is not important and hence removed
        """

        def to_one_hot(array, channel_axis, num_classes=3):
            """
            Changes class information into one-hot encoded information
            Number of classes in KiTS19 is 3: background, kidney segmentation, tumor segmentation
            As a result, 1 channel of class info turns into 3 channels of one-hot info
            """
            res = np.eye(num_classes)[np.array(array).reshape(-1)]
            array = res.reshape(list(array.shape) + [num_classes])
            return np.transpose(array, (0, 4, 1, 2, 3)).astype(np.float64)

        # constants
        channel_axis = 1
        reduce_axis = (2, 3, 4)
        smooth_nr = 1e-6
        smooth_dr = 1e-6

        # apply one-hot
        prediction = to_one_hot(prediction, channel_axis)
        target = to_one_hot(target, channel_axis)

        # remove background
        target = target[:, 1:]
        prediction = prediction[:, 1:]

        # calculate dice score
        assert target.shape == prediction.shape, \
            f"Different shape -- target: {target.shape}, prediction: {prediction.shape}"
        assert target.dtype == np.float64 and prediction.dtype == np.float64, \
            f"Unexpected dtype -- target: {target.dtype}, prediction: {prediction.dtype}"

        # intersection for numerator; target/prediction sum for denominator
        # easy b/c one-hot encoded format
        intersection = np.sum(target * prediction, axis=reduce_axis)
        target_sum = np.sum(target, axis=reduce_axis)
        prediction_sum = np.sum(prediction, axis=reduce_axis)

        # get DICE score for each class
        dice_val = (2.0 * intersection + smooth_nr) / (target_sum + prediction_sum + smooth_dr)
        self.__kidney_score += dice_val[0][0]
        self.__tumor_score += dice_val[0][1]

    def submit_predictions(self, prediction):
        self.__current_image.accumulate_result_slice(prediction)
        if self.__current_image.all_issued:
            full_prediction = self.__current_image.get_final_result()
            ground_truth = np.expand_dims(nib.load(self.__get_gt_path()).get_fdata().astype(np.uint8), axis=0)
            self.__calc_dice_score(full_prediction, ground_truth)
            self.__current_img_id += 1

    def summarize_accuracy(self):
        if self.__current_img_id < 1:
            utils.print_warning_message(
                "Not a single image has been completed - cannot calculate accuracy. Note that images of KiTS dataset "
                "are processed in slices due to their size. That implies that complete processing of one image can "
                "involve many passes through the network.")
            return {"mean_kidney_acc": None, "mean_tumor_acc": None, "mean_composite_acc": None}

        mean_kidney = self.__kidney_score / self.__current_img_id
        #print("\n Mean kidney segmentation accuracy = {:.3f}".format(mean_kidney))

        mean_tumor = self.__tumor_score / self.__current_img_id
        #print(" Mean tumor segmentation accuracy = {:.3f}".format(mean_tumor))

        mean_composite = (mean_kidney + mean_tumor) / 2
        #print(" Mean composite accuracy = {:.3f}".format(mean_composite))

        #print(f"\nAccuracy figures above calculated on the basis of {self.__current_img_id} images.")
        return {"mean_kidney_acc": mean_kidney, "mean_tumor_acc": mean_tumor, "mean_composite_acc": mean_composite}
