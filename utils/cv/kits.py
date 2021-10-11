import pickle
import pathlib
import numpy as np
import pandas as pd
import nibabel as nib
from pathlib import Path
from multiprocessing import Pool

import utils.cv.dataset as utils_ds
import utils.cv.pre_processing as pre_p
from utils.cv.kits_preprocessing import preprocess_with_multiproc
from utils.cv.kits_preprocessing import ROI_SHAPE, SLIDE_OVERLAP_FACTOR
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
        self.__preprocessed_files_pkl_path = Path(self.__dataset_dir_path, "preprocessed_files.pkl")
        self.__loaded_files = {}
        self.__current_img_id = 0
        self.__current_image = self.__Image()

        if not self.__preprocessed_files_pkl_path.exists():
            self.__preprocess()
        print(pickle.load(open(self.__preprocessed_files_pkl_path, "rb")))
        self.__file_names = pickle.load(open(self.__preprocessed_files_pkl_path, "rb"))["file_list"]

        print(self.__file_names)
        self.__file_name = None
        self.__bundle = list()
        self.__dice_scores = None
        self.__current_file_name = None
        self.available_instances = len(self.__file_names)

        super().__init__()

    def __preprocess(self):
        class args_substitute:
            def __init__(self, raw_data_dir, results_dir, num_proc=4):
                self.data_dir = raw_data_dir
                self.results_dir = results_dir
                self.calibration = False
                self.num_proc = num_proc
        preprocess_with_multiproc(args_substitute(self.__dataset_dir_path, self.__dataset_dir_path))

    def __get_path_to_img(self):
        """
        A function providing path to the KiTS19 image.

        :return: pathlib.PurePath object containing path to the image
        """
        try:
            file_name = self.__file_names[self.__current_img_id]
        except IndexError:
            raise utils.OutOfInstances("No more KiTS19 images to process in the directory provided")
        self.__current_img_id += 1
        return pathlib.PurePath(self.__dataset_dir_path, f"{file_name}.pkl")

    class __Image:

        def __init__(self):
            self.__full_image = None
            self.all_issued = False
            self.empty = True
            self.__slice_indices = None
            self.__current_slice_id = None

        def assign(self, image):
            self.__full_image = image
            self.all_issued = False
            self.empty = False
            self.__slice_indices = list()
            self.__current_slice_id = 0

            image_shape = image.shape[1:]
            dims = len(image_shape)
            strides = [int(ROI_SHAPE[i] * (1 - SLIDE_OVERLAP_FACTOR)) for i in range(dims)]
            size = [(image_shape[i] - ROI_SHAPE[i]) // strides[i] + 1 for i in range(dims)]

            for i in range(0, strides[0] * size[0], strides[0]):
                for j in range(0, strides[1] * size[1], strides[1]):
                    for k in range(0, strides[2] * size[2], strides[2]):
                        self.__slice_indices.append((ROI_SHAPE[0] + i, ROI_SHAPE[1] + j, ROI_SHAPE[2] + k))

        def get_next_slice(self):
            assert self.all_issued is False and self.empty is False
            slice = self.__full_image[-1, *self.__slice_indices[self.__current_slice_id]]
            self.__current_slice_id += 1
            if self.__current_slice_id == len(self.__slice_indices):
                self.all_issued = True
            return slice


    def get_input_array(self):
        """
        A function returning an array containing pre-processed image, a result array, a norm_map and norm_patch.
        """
        if self.__current_image.all_issued or self.__current_image.empty:
            self.__current_image.assign(pickle.load(open(self.__get_path_to_img(), "rb"))[0])
        return self.__current_image.get_next_slice()


        # for i, j, k in self.__get_slice_for_sliding_window(img, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
        #     input_slice = image[
        #                   ...,
        #                   i:(ROI_SHAPE[0] + i),
        #                   j:(ROI_SHAPE[1] + j),
        #                   k:(ROI_SHAPE[2] + k)]

    def __get_slice_for_sliding_window(self, image, roi_shape=ROI_SHAPE, overlap=SLIDE_OVERLAP_FACTOR):

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
        # note: layout is assumed to be linear (NCDHW) always
        image = apply_norm_map(image, norm_map)
        image = apply_argmax(image)

        return image

    def submit_predictions(self, prediction):
        """
        Collects and summarizes DICE scores of all the predicted files using multi-processes
        """
        path_to_groundtruth = Path(self.__groundtruth_path, self.__file_name, 'segmentation.nii.gz')
        groundtruth = nib.load(path_to_groundtruth).get_fdata().astype(np.uint8)

        groundtruth = np.expand_dims(groundtruth, 0)
        prediction = np.expand_dims(prediction, 0)

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
