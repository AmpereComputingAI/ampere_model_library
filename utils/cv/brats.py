# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2020 Division of Medical Image Computing, German Cancer Research Center (DKFZ), Heidelberg, Germany
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pickle
import sys
import shutil
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from collections import OrderedDict
from batchgenerators.utilities.file_and_folder_operations import subdirs, isfile, subfiles, maybe_mkdir_p, save_json
from batchgenerators.augmentations.utils import pad_nd_image

import utils.misc as utils
from utils.helpers import Dataset
utils_cv_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "nnUNet")
sys.path.append(utils_cv_path)


class BraTS19(Dataset):
    """
    A class providing facilities for preprocessing and postprocessing of BraTS 2019 dataset.
    """

    def __init__(self, dataset_dir_path=None):

        if dataset_dir_path is None:
            env_var = "BRATS19_DATASET_PATH"
            dataset_dir_path = utils.get_env_variable(
                env_var, f"Path to BraTS19 dataset directory has not been specified with {env_var} flag")

        self.__dataset_dir_path = dataset_dir_path
        self.__preprocessed_dir_path = Path(self.__dataset_dir_path, "preprocessed")

        self.__input_details = list()

        if not self.__preprocessed_dir_path.is_dir():
            self.__preprocess()

        with open(os.path.join(Path(self.__preprocessed_dir_path, "preprocessed_files.pkl")), "rb") as f:
            self.__input_file_names = pickle.load(f)

        self.__current_img_id = 0
        self.available_instances = len(self.__input_file_names)

        self.__processed_predictions_dir_path = Path("/tmp/processed_predictions")
        if self.__processed_predictions_dir_path.is_dir():
            shutil.rmtree(self.__processed_predictions_dir_path)
        self.__processed_predictions_dir_path.mkdir()

    def __convert_data(self):
        def copy_BraTS_segmentation_and_convert_labels(in_file, out_file):
            # use this for segmentation only!!!
            # nnUNet wants the labels to be continuous. BraTS is 0, 1, 2, 4 -> we make that into 0, 1, 2, 3
            img = sitk.ReadImage(str(in_file))
            img_npy = sitk.GetArrayFromImage(img)

            uniques = np.unique(img_npy)
            for u in uniques:
                if u not in [0, 1, 2, 4]:
                    raise RuntimeError('unexpected label')

            seg_new = np.zeros_like(img_npy)
            seg_new[img_npy == 4] = 3
            seg_new[img_npy == 2] = 1
            seg_new[img_npy == 1] = 2
            img_corr = sitk.GetImageFromArray(seg_new)
            img_corr.CopyInformation(img)
            sitk.WriteImage(img_corr, str(out_file))

        target_base = Path("/tmp/Task043_BraTS2019")

        target_imagesTr = Path(target_base, "imagesTr")
        target_labelsTr = Path(self.__preprocessed_dir_path, "labelsTr")

        maybe_mkdir_p(target_base)
        maybe_mkdir_p(target_imagesTr)
        maybe_mkdir_p(target_labelsTr)

        patient_names = []
        for tpe in ["HGG", "LGG"]:
            cur = Path(self.__dataset_dir_path, "MICCAI_BraTS_2019_Data_Training", tpe)
            for p in subdirs(cur, join=False):
                patient_dir = Path(cur, p)
                patient_name = f"{tpe}__{p}"
                patient_names.append(patient_name)
                t1 = Path(patient_dir, f"{p}_t1.nii.gz")
                t1c = Path(patient_dir, f"{p}_t1ce.nii.gz")
                t2 = Path(patient_dir, f"{p}_t2.nii.gz")
                flair = Path(patient_dir, f"{p}_flair.nii.gz")
                seg = Path(patient_dir, f"{p}_seg.nii.gz")

                assert all([
                    isfile(t1),
                    isfile(t1c),
                    isfile(t2),
                    isfile(flair),
                    isfile(seg)
                ]), "%s" % patient_name

                shutil.copy(t1, Path(target_imagesTr, f"{patient_name}_0000.nii.gz"))
                shutil.copy(t1c, Path(target_imagesTr, f"{patient_name}_0001.nii.gz"))
                shutil.copy(t2, Path(target_imagesTr, f"{patient_name}_0002.nii.gz"))
                shutil.copy(flair, Path(target_imagesTr, f"{patient_name}_0003.nii.gz"))

                copy_BraTS_segmentation_and_convert_labels(seg, Path(target_labelsTr, f"{patient_name}.nii.gz"))

        json_dict = OrderedDict()
        json_dict['name'] = "BraTS2019"
        json_dict['description'] = "nothing"
        json_dict['tensorImageSize'] = "4D"
        json_dict['reference'] = "see BraTS2019"
        json_dict['licence'] = "see BraTS2019 license"
        json_dict['release'] = "0.0"
        json_dict['modality'] = {
            "0": "T1",
            "1": "T1ce",
            "2": "T2",
            "3": "FLAIR"
        }
        json_dict['labels'] = {
            "0": "background",
            "1": "edema",
            "2": "non-enhancing",
            "3": "enhancing",
        }
        json_dict['numTraining'] = len(patient_names)
        json_dict['numTest'] = 0
        json_dict['training'] = [
            {'image': "./imagesTr/%s.nii.gz" % i, "label": "./labelsTr/%s.nii.gz" % i} for i in patient_names
        ]
        json_dict['test'] = []

        save_json(json_dict, Path(target_base, "dataset.json"))
        return target_imagesTr

    def __preprocess(self):
        """
        Function delegating the offline pre-processing work to slightly modified MLCommons script.
        """
        from utils.cv.nnUNet.nnunet.inference.predict import preprocess_multithreaded
        from utils.cv.nnUNet.nnunet.training.model_restore import load_model_and_checkpoint_files

        validation_files = list()
        with open(Path(self.__dataset_dir_path, "fold1_validation.txt")) as f:
            for line in f:
                validation_files.append(line.rstrip())

        raw_data_dir_path = self.__convert_data()

        all_files = subfiles(raw_data_dir_path, suffix=".nii.gz", join=False, sort=True)
        list_of_lists = [[str(Path(raw_data_dir_path, i)) for i in all_files if i[:len(j)].startswith(j) and
                          len(i) == (len(j) + 12)] for j in validation_files]

        trainer, _ = load_model_and_checkpoint_files(
            str(Path(self.__dataset_dir_path, "nnUNetTrainerV2__nnUNetPlansv2.mlperf.1")),
            folds=1,
            checkpoint_name="model_final_checkpoint"
        )
        preprocessed_iterator = preprocess_multithreaded(trainer, list_of_lists, validation_files, num_processes=4)

        all_output_files = []
        for preprocessed in preprocessed_iterator:
            output_filename, (d, dct) = preprocessed

            all_output_files.append(output_filename)
            if isinstance(d, str):
                data = np.load(d)
                os.remove(d)
                d = data

            # Pad to the desired full volume
            d = pad_nd_image(d, trainer.patch_size, "constant", None, False, None)

            with open(Path(self.__preprocessed_dir_path, output_filename + ".pkl"), "wb") as f:
                pickle.dump([d, dct], f)
            f.close()
        with open(Path(self.__preprocessed_dir_path, "preprocessed_files.pkl"), "wb") as f:
            pickle.dump(all_output_files, f)

    def reset(self):
        self.__current_img_id = 0
        self.__processed_predictions_dir_path = Path("/tmp/processed_predictions")
        if self.__processed_predictions_dir_path.is_dir():
            shutil.rmtree(self.__processed_predictions_dir_path)
        self.__processed_predictions_dir_path.mkdir()
        return True

    def get_input_array(self):
        """
        A function returning an array containing slice of pre-processed image.
        """
        try:
            data = pickle.load(
                open(Path(self.__preprocessed_dir_path, f"{self.__input_file_names[self.__current_img_id]}.pkl"), "rb"))
        except IndexError:
            raise utils.OutOfInstances("No more BraTS19 images to process in the directory provided")
        self.__input_details.append(data[1])
        return data[0]

    def __pad_prediction(self, result):
        padded_shape = [224, 224, 160]
        raw_shape = list(self.__input_details[self.__current_img_id]["size_after_cropping"])
        # Remove the padded part
        pad_before = [(p - r) // 2 for p, r in zip(padded_shape, raw_shape)]
        pad_after = [-(p - r - b) for p, r, b in zip(padded_shape, raw_shape, pad_before)]
        result_shape = (4,) + tuple(padded_shape)
        result = result.reshape(result_shape).astype(np.float16)
        return result[:, pad_before[0]:pad_after[0], pad_before[1]:pad_after[1], pad_before[2]:pad_after[2]]

    def submit_predictions(self, prediction):
        prediction = self.__pad_prediction(prediction)
        from utils.cv.nnUNet.nnunet.inference.segmentation_export import save_segmentation_nifti_from_softmax
        save_segmentation_nifti_from_softmax(
            prediction,
            str(Path(self.__processed_predictions_dir_path,
                     f"{self.__input_file_names[self.__current_img_id]}.nii.gz")),
            self.__input_details[self.__current_img_id],
            order=3,
            verbose=False
        )
        self.__current_img_id += 1

    def _summarize_accuracy(self):
        from utils.cv.nnUNet.nnunet.evaluation.region_based_evaluation import evaluate_regions, get_brats_regions
        evaluate_regions(
            str(self.__processed_predictions_dir_path),
            str(Path(self.__preprocessed_dir_path, "labelsTr")),
            get_brats_regions()
        )
        with open(Path(self.__processed_predictions_dir_path, "summary.csv")) as f:
            for line in f:
                words = line.split(",")
                if words[0] == "mean":
                    whole_tumor = float(words[1])
                    tumor_core = float(words[2])
                    enhancing_tumor = float(words[3])
                    mean_composite = (whole_tumor + tumor_core + enhancing_tumor) / 3
                    break

        # print("\n Mean whole tumor segmentation accuracy = {:.3f}".format(whole_tumor))
        # print(" Mean tumor core segmentation accuracy = {:.3f}".format(tumor_core))
        # print(" Mean enhancing tumor segmentation accuracy = {:.3f}".format(enhancing_tumor))
        # print(" Mean composite accuracy = {:.3f}".format(mean_composite))
        #
        # print(f"\nAccuracy figures above calculated on the basis of {self.__current_img_id} images.")
        return {
            "mean_whole_tumor_acc": whole_tumor,
            "mean_tumor_core_acc": tumor_core,
            "mean_enhancing_tumor_acc": enhancing_tumor,
            "mean_composite_acc": mean_composite
        }
