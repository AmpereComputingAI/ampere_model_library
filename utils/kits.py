import utils.cv.dataset as utils_ds
import utils.cv.pre_processing as pre_p
from utils.global_vars import ROI_SHAPE, SLIDE_OVERLAP_FACTOR
from utils.unet_preprocessing import get_dice_score, apply_norm_map, apply_argmax

import pickle
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import nibabel as nib
import pandas as pd
import pathlib

GROUNDTRUTH_PATH = '/onspecta/dev/mz/temp/datasets/kits19_preprocessed/nifti/case_00000/segmentation.nii.gz'
GROUNDTRUTH_PATH_GRAVITON = '/onspecta/mz/temp/datasets/kits19_preprocessed/nifti/case_00000/segmentation.nii.gz'

POSTPROCESSED_DIR = '/onspecta/dev/mz/unet_results'
POSTPROCESSED_DIR_GRAVITON = '/onspecta/mz/unet_results'
CASE = 'case_000000'


class KiTS19(utils_ds.ImageDataset):
    """
    A class providing facilities for preprocessing of KiTS19 dataset.
    """

    def __init__(self, images_path=None, images_anno=None, groundtruth_path=None):

        if images_path is None:
            env_var = "KITS19_IMAGES_DIR"
            images_path = utils.get_env_variable(
                env_var, f"Path to KiTS19 images directory has not been specified with {env_var} flag")

        if images_anno is None:
            env_var = "KITS19_PREPROCESSED_FILE_PATH"
            images_anno = utils.get_env_variable(
                env_var, f"Path to KiTS19 preprocessed_file.pkl has not been specified with {env_var} flag")

        if groundtruth_path is None:
            env_var = "KITS19_GROUNDTRUTH_PATH"
            images_anno = utils.get_env_variable(
                env_var, f"Path to KiTS19 nifti folder has not been specified with {env_var} flag")

        self.__images_path = images_path
        self.__images_anno = images_anno
        self.__groundtruth_path = groundtruth_path
        self.__loaded_files = {}
        self.__current_img = 0
        self.__file_names = self.__deserialize_file()
        self.__file_name = None
        self.__bundle = list()
        self.__dice_scores = None
        self.__current_file_name = None
        self.available_instances = len(self.__file_names)

        super().__init__()

    def __deserialize_file(self):
        """
        A function deserializing pickle file containing name of the images in KiTS19 dataset.

        :return: list of strings
        """
        with open(Path(self.__images_anno), "rb") as f:
            return pickle.load(f)['file_list']

    def __get_path_to_img(self):
        """
        A function providing path to the KiTS19 image.

        :return: pathlib.PurePath object containing path to the image
        """
        try:
            self.__file_name = self.__file_names[self.__current_img]

        except IndexError:
            raise utils_ds.OutOfInstances("No more images to process in the directory provided")
        self.__current_img += 1

        return pathlib.PurePath(self.__images_path, self.__file_name + '.pkl')

    def get_input_array(self):
        """
        A function returning an array containing pre-processed image, a result array, a norm_map and norm_patch.

        :return: numpy array containing rescaled, pre-processed image data of batch size requested at class
        :return: numpy array containing rescaled, pre-processed image data of batch size requested at class
        :return: numpy array containing rescaled, pre-processed image data of batch size requested at class
        :return: numpy array containing rescaled, pre-processed image data of batch size requested at class
        initialization
        """
        # file_name = self.__file_names[self.__current_img]
        self.__current_file_name = self.__get_path_to_img()
        print(self.__current_file_name)
        with open(Path(self.__current_file_name), "rb") as f:
            self.__loaded_files[self.__current_img] = pickle.load(f)[0]

        image = self.__loaded_files[self.__current_img][np.newaxis, ...]
        result, norm_map, norm_patch = self.prepare_arrays(image)

        return image, result, norm_map, norm_patch

    def get_slice_for_sliding_window(self, image, roi_shape=ROI_SHAPE, overlap=SLIDE_OVERLAP_FACTOR):

        assert len(roi_shape) == 3 and any(roi_shape) and all(dim > 0 for dim in roi_shape), \
            f"Need proper ROI shape! The current ROI shape is: {roi_shape}"

        assert 0 < overlap < 1, \
            f"Need sliding window overlap factor in (0,1)! The current overlap factor is: {overlap}"

        image_shape = image.shape[2:]
        dim = len(image_shape)
        strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]
        size = [(image_shape[i] - roi_shape[i]) // strides[i] + 1 for i in range(dim)]

        for i in range(0, strides[0] * size[0], strides[0]):
            for j in range(0, strides[1] * size[1], strides[1]):
                for k in range(0, strides[2] * size[2], strides[2]):
                    yield i, j, k

    def prepare_arrays(self, image, roi_shape=ROI_SHAPE):

        assert isinstance(roi_shape, list) and len(roi_shape) == 3 and any(roi_shape), \
            f"Need proper ROI shape: {roi_shape}"

        image_shape = list(image.shape[2:])
        result = np.zeros(shape=(1, 3, *image_shape), dtype=image.dtype)

        norm_map = np.zeros_like(result)

        norm_patch = pre_p.gaussian_kernel(
            roi_shape[0], 0.125 * roi_shape[0]).astype(norm_map.dtype)

        return result, norm_map, norm_patch

    def finalize(self, image, norm_map):
        """
        Finalizes results obtained from sliding window inference
        """
        # NOTE: layout is assumed to be linear (NCDHW) always
        # apply norm_map
        image = apply_norm_map(image, norm_map)

        # argmax
        image = apply_argmax(image)

        return image

    def submit_predictions(self, prediction):
        """
        Collects and summarizes DICE scores of all the predicted files using multi-processes
        """
        path_to_groundtruth = Path(self.__groundtruth_path, self.__file_name, 'segmentation.nii.gz')

        print(path_to_groundtruth)
        groundtruth = nib.load(path_to_groundtruth).get_fdata().astype(np.uint8)

        groundtruth = np.expand_dims(groundtruth, 0)
        prediction = np.expand_dims(prediction, 0)

        assert groundtruth.shape == prediction.shape, \
            "{} -- groundtruth: {} and prediction: {} have different shapes".format(
                self.__file_name, groundtruth.shape, prediction.shape)

        self.__bundle.append((self.__file_name, groundtruth, prediction))

        with Pool(1) as p:
            self.__dice_scores = p.starmap(get_dice_score, self.__bundle)

        # self.summarize_accuracy_test(dice_scores)

    #     self.save_evaluation_summary(POSTPROCESSED_DIR_GRAVITON, dice_scores)
    #
    # def save_evaluation_summary(self, POSTPROCESSED_DIR_GRAVITON, dice_scores):
    #     """
    #     Stores collected DICE scores in CSV format: $(POSTPROCESSED_DATA_DIR)/summary.csv
    #     """
    #     sum_path = Path(POSTPROCESSED_DIR_GRAVITON, "summary.csv").absolute()
    #     df = pd.DataFrame()
    #
    #     print(sum_path)
    #
    #     for _s in dice_scores:
    #         case, arr = _s
    #         kidney = arr[0]
    #         tumor = arr[1]
    #         composite = np.mean(arr)
    #         df = df.append(
    #             {
    #                 "case": case,
    #                 "kidney": kidney,
    #                 "tumor": tumor,
    #                 "composite": composite
    #             }, ignore_index=True)
    #
    #     df.set_index("case", inplace=True)
    #     # consider NaN as a crash hence zero
    #     df.loc["mean"] = df.fillna(0).mean()
    #
    #     print(df)
    #
    #     df.to_csv(sum_path)

    def summarize_accuracy(self):
        with Pool(1) as p:
            self.__dice_scores = p.starmap(get_dice_score, self.__bundle)

        sum_path = Path(POSTPROCESSED_DIR_GRAVITON, "summary.csv").absolute()
        df = pd.DataFrame()

        print(sum_path)

        for _s in dice_scores:
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
        # consider NaN as a crash hence zero
        df.loc["mean"] = df.fillna(0).mean()

        print(df)

        df.to_csv(sum_path)

        with open(Path(POSTPROCESSED_DIR_GRAVITON, "summary.csv")) as f:
            for line in f:
                if not line.startswith("mean"):
                    continue
                words = line.split(",")
                if words[0] == "mean":
                    composite = float(words[1])
                    kidney = float(words[2])
                    tumor = float(words[3])
                    print("Accuracy: mean = {:.5f}, kidney = {:.4f}, tumor = {:.4f}".format(
                        composite, kidney, tumor))
                    break

        print(len(self.__bundle))
        print(self.__bundle[0][0])
        print(self.__bundle[1][0])
        print(type(self.__dice_scores))
        print(self.__dice_scores)

    # def summarize_accuracy(self):
        # with open(Path(POSTPROCESSED_DIR_GRAVITON, "summary.csv")) as f:
        #     for line in f:
        #         if not line.startswith("mean"):
        #             continue
        #         words = line.split(",")
        #         if words[0] == "mean":
        #             composite = float(words[1])
        #             kidney = float(words[2])
        #             tumor = float(words[3])
        #             print("Accuracy: mean = {:.5f}, kidney = {:.4f}, tumor = {:.4f}".format(
        #                 composite, kidney, tumor))
        #             break

        #
        # pass