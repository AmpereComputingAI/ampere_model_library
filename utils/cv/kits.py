import pathlib
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from multiprocessing import Pool
from scipy import signal

import utils.cv.dataset as utils_ds
import utils.cv.pre_processing as pre_p
from utils.cv.kits_preprocessing import preprocess_with_multiproc, ROI_SHAPE, SLIDE_OVERLAP_FACTOR, TARGET_CASES
from utils.unet_preprocessing import get_dice_score, apply_norm_map, apply_argmax


class KiTS19(utils_ds.ImageDataset):
    """
    A class providing facilities for preprocessing of KiTS19 dataset.
    """

    def __init__(self, dataset_dir_path=None):

        if dataset_dir_path is None:
            env_var = "KITS19_DATASET_PATH"
            dataset_dir_path = utils.get_env_variable(
                env_var, f"Path to KiTS19 dataset directory has not been specified with {env_var} flag")

        self.__dataset_dir_path = dataset_dir_path
        self.__preprocessed_files_dir_path = Path(self.__dataset_dir_path, "preprocessed")
        self.__loaded_files = {}
        self.__current_img_id = 0
        self.__current_image = self.__Image()

        if not self.__preprocessed_files_dir_path.is_dir():
            self.__preprocess()
        self.__case_ids = TARGET_CASES
        self.__case_ids.sort()

        self.__file_name = None
        self.__bundle = list()
        self.__dice_scores = None
        self.__current_file_name = None
        self.available_instances = len(self.__case_ids)

        super().__init__()

    def __preprocess(self):
        class args_substitute:
            def __init__(self, raw_data_dir, results_dir, num_proc=4):
                self.data_dir = raw_data_dir
                self.results_dir = results_dir
                self.mode = "preprocess"
                self.calibration = False
                self.num_proc = num_proc

        args = args_substitute(self.__dataset_dir_path, self.__dataset_dir_path)
        preprocess_with_multiproc(args)

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

        def __init__(self):
            # self.__full_image = None
            self.all_issued = False
            self.empty = True
            self.__norm_patch = self.__gen_norm_patch()
            # self.__slice_indices = None
            # self.__current_slice_id = None
            # self.__current_result_slice = None
            # self.result = None

        def __gen_norm_patch(self, std_factor=0.125):
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
            result_slice = self.__result[
                           ...,
                           i:(ROI_SHAPE[0] + i),
                           j:(ROI_SHAPE[1] + j),
                           k:(ROI_SHAPE[2] + k)
                           ]
            result_slice += output * self.__norm_patch
            a = output * self.__norm_patch
            import hashlib
            print(hashlib.md5(a.numpy().tostring()).hexdigest())
            fddf
            self.__current_slice_id += 1
            if self.__current_slice_id == len(self.__slice_indices):
                self.all_issued = True

        def get_final_result(self):
            self.__result /= self.__norm_map
            return np.argmax(self.__result, axis=1).astype(np.uint8)

    def get_input_array(self):
        """
        A function returning an array containing pre-processed image, a result array, a norm_map and norm_patch.
        """
        if self.__current_image.all_issued or self.__current_image.empty:
            print(nib.load(self.__get_path_to_img()).get_fdata())
            print(self.__get_path_to_img())
            self.__current_image.assign(nib.load(self.__get_path_to_img()).get_fdata())
        return self.__current_image.get_next_input_slice()

    def __get_gt_path(self):
        try:
            case_id = self.__case_ids[self.__current_img_id]
        except IndexError:
            raise utils.OutOfInstances("No more KiTS19 images to process in the directory provided")
        return pathlib.PurePath(self.__preprocessed_files_dir_path, case_id, "segmentation.nii.gz")

    def submit_predictions(self, prediction):
        """
        Collects and summarizes DICE scores of all the predicted files using multi-processes
        """
        self.__current_image.accumulate_result_slice(prediction)
        if self.__current_image.all_issued:
        #if True:
            full_prediction = self.__current_image.get_final_result()
            ground_truth = np.expand_dims(nib.load(self.__get_gt_path()).get_fdata().astype(np.uint8), axis=0)
            self.__current_img_id += 1
            print(full_prediction.shape)
            print(ground_truth.shape)
            print(get_dice_score("0000", full_prediction, ground_truth))
            ds

            #groundtruth = np.expand_dims(groundtruth, 0)


            assert groundtruth.shape == prediction.shape, \
                "{} -- groundtruth: {} and prediction: {} have different shapes".format(
                    self.__file_name, groundtruth.shape, prediction.shape)

            self.__bundle.append((self.__file_name, groundtruth, prediction))

    def summarize_accuracy(self):
        with Pool(1) as p:
            self.__dice_scores = p.starmap(get_dice_score, self.__bundle)

        df = pd.DataFrame()

        for _s in self.__dice_scores:
            case, arr = _s
            kidney = arr[0]
            tumor = arr[1]
            composite = np.mean(arr)
            df = df.append(
                {
                    "case": case,
                    "kidney": kidney,
                    "tumor": tumor,
                    "composite": composite
                }, ignore_index=True)

        df.set_index("case", inplace=True)
        df.loc["mean"] = df.fillna(0).mean()

        mean_composite = df.loc['mean', 'composite']
        mean_kidney = df.loc['mean', 'kidney']
        mean_tumor = df.loc['mean', 'tumor']

        print(f" mean composite {mean_composite}")
        print(f" mean kidney {mean_kidney}")
        print(f" mean composite {mean_tumor}")

        return {"mean_composite": mean_composite, "mean_kidney": mean_kidney, "mean_tumor": mean_tumor}
