import utils.cv.dataset as utils_ds
import utils.cv.pre_processing as pre_p
from utils.global_vars import ROI_SHAPE, SLIDE_OVERLAP_FACTOR

import pickle
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import nibabel as nib

GROUNDTRUTH_PATH = '/onspecta/dev/mz/temp/datasets/kits19_preprocessed/nifti/case_00000/segmentation.nii.gz'
GROUNDTRUTH_PATH_GRAVITON = '/onspecta/dev/mz/temp/datasets/kits19_preprocessed/nifti/case_00000/segmentation.nii.gz'
CASE = 'case_000000'

class KiTS19(utils_ds.ImageDataset):
    """
    A class providing facilities for preprocessing of KiTS19 dataset.
    """

    def __init__(self, images_path=None, images_anno=None):

        if images_path is None:
            env_var = "KITS19_IMAGES_DIR"
            images_path = utils.get_env_variable(
                env_var, f"Path to KiTS19 images directory has not been specified with {env_var} flag")

        if images_anno is None:
            env_var = "KITS19_PREPROCESSED_FILE_PATH"
            images_anno = utils.get_env_variable(
                env_var, f"Path to KiTS19 preprocessed_file.pkl has not been specified with {env_var} flag")

        self.__images_path = images_path
        self.__images_anno = images_anno
        self.__loaded_files = {}
        self.__current_img = 0
        self.__file_names = self.__deserialize_file()
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
            file_name = self.__file_names[self.__current_img]
        except IndexError:
            raise utils_ds.OutOfInstances("No more images to process in the directory provided")
        self.__current_img += 1
        return pathlib.PurePath(self.__images_path, file_name, '.pkl')

    def get_input_array(self):
        """
        A function returning an array containing pre-processed image, a result array, a norm_map and norm_patch.

        :return: numpy array containing rescaled, pre-processed image data of batch size requested at class
        :return: numpy array containing rescaled, pre-processed image data of batch size requested at class
        :return: numpy array containing rescaled, pre-processed image data of batch size requested at class
        :return: numpy array containing rescaled, pre-processed image data of batch size requested at class
        initialization
        """
        file_name = self.__file_names[self.__current_img]
        with open(Path(self.__images_path, "{:}.pkl".format(file_name)), "rb") as f:
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
        image = self.apply_norm_map(image, norm_map)

        # argmax
        image = apply_argmax(image)

        return image

    def apply_norm_map(self, image, norm_map):
        """
        Applies normal map norm_map to image and return the outcome
        """
        image /= norm_map
        return image

    def apply_argmax(self, image):
        """
        Returns indices of the maximum values along the channel axis
        Input shape is (bs=1, channel=3, (ROI_SHAPE)), float -- sub-volume inference result
        Output shape is (bs=1, channel=1, (ROI_SHAPE)), integer -- segmentation result
        """
        channel_axis = 1
        image = np.argmax(image, axis=channel_axis).astype(np.uint8)
        image = np.expand_dims(image, axis=0)

        return image

    def prepare_one_hot(self, my_array, num_classes):
        """
        Reinterprets my_array into one-hot encoded, for classes as many as num_classes
        """
        res = np.eye(num_classes)[np.array(my_array).reshape(-1)]
        return res.reshape(list(my_array.shape) + [num_classes])

    def to_one_hot(self, my_array, channel_axis):
        """
        Changes class information into one-hot encoded information
        Number of classes in KiTS19 is 3: background, kidney segmentation, tumor segmentation
        As a result, 1 channel of class info turns into 3 channels of one-hot info
        """
        my_array = self.prepare_one_hot(my_array, num_classes=3)
        my_array = np.transpose(my_array, (0, 4, 1, 2, 3)).astype(np.float64)
        return my_array

    def get_dice_score(self, case, prediction, target):
        """
        Calculates DICE score of prediction against target, for classes as many as case
        One-hot encoded form of case/prediction used for easier handling
        Background case is not important and hence removed
        """
        # constants
        channel_axis = 1
        reduce_axis = (2, 3, 4)
        smooth_nr = 1e-6
        smooth_dr = 1e-6

        # apply one-hot
        prediction = self.to_one_hot(prediction, channel_axis)
        target = self.to_one_hot(target, channel_axis)

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
        dice_val = (2.0 * intersection + smooth_nr) / \
                   (target_sum + prediction_sum + smooth_dr)

        # return after removing batch dim
        print(case, dice_val[0])
        return (case, dice_val[0])

    def submit_predictions(self, prediction):
        """
        Collects and summarizes DICE scores of all the predicted files using multi-processes
        """
        bundle = list()

        groundtruth = nib.load(GROUNDTRUTH_PATH_GRAVITON).get_fdata().astype(np.uint8)

        bundle.append((CASE, groundtruth, prediction))

        # for case in target_files:
        #     groundtruth_path = Path(preprocessed_data_dir,
        #                             "nifti", case, "segmentation.nii.gz").absolute()
        #     prediction_path = Path(postprocessed_data_dir,
        #                            case, "prediction.nii.gz").absolute()
        #
        #     groundtruth = nib.load(groundtruth_path).get_fdata().astype(np.uint8)
        #     prediction = nib.load(prediction_path).get_fdata().astype(np.uint8)
        #
        #     groundtruth = np.expand_dims(groundtruth, 0)
        #     prediction = np.expand_dims(prediction, 0)
        #
        #     assert groundtruth.shape == prediction.shape, \
        #         "{} -- groundtruth: {} and prediction: {} have different shapes".format(
        #             case, groundtruth.shape, prediction.shape)
        #
        #     bundle.append((case, groundtruth, prediction))

        with Pool(1) as p:
            dice_scores = p.starmap(self.get_dice_score, bundle)

        save_evaluation_summary(postprocessed_data_dir, dice_scores)

    def summarize_accuracy(self):
        # TODO: implement this
        pass
