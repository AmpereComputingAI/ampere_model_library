import utils.dataset as utils_ds
import pickle
from pathlib import Path
import numpy as np
import utils.pre_processing as pre_p

# SAMPLE_LIST = [14, 32, 33, 23, 25, 31, 0, 5, 39, 21, 9, 19, 29, 38, 20, 30]
ROI_SHAPE = [128, 128, 128]
SLIDE_OVERLAP_FACTOR = 0.5


class Kits(utils_ds.ImageDataset):
    """
    A class providing facilities for preprocessing and postprocessing of ImageNet validation dataset.
    """

    def __init__(self, images_path=None, images_anno=None):

        if images_path is None:
            env_var = "IMAGENET_IMG_PATH"
            images_path = utils.get_env_variable(
                env_var, f"Path to ImageNet images directory has not been specified with {env_var} flag")

        if images_anno is None:
            env_var = "IMAGENET_IMG_PATH"
            images_path = utils.get_env_variable(
                env_var, f"Path to ImageNet images directory has not been specified with {env_var} flag")

        self.__images_path = images_path
        self.__images_anno = images_anno
        self.__loaded_files = {}
        self.__current_img = 0
        self.__file_names = self.parse_file()
        self.available_instances = len(self.__file_names)
        super().__init__()

    def parse_file(self):
        with open(Path(self.__images_anno), "rb") as f:
            return pickle.load(f)['file_list']

    def __get_path_to_img(self):
        """
        A function providing path to the ImageNet image.

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
        A function returning an array containing pre-processed rescaled image's or multiple images' data.

        :param target_shape: tuple of intended image shape (height, width)
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
        """
        Returns indices for image stride, to fulfill sliding window inference
        Stride is determined by roi_shape and overlap
        """
        assert isinstance(roi_shape, list) and len(roi_shape) == 3 and any(roi_shape), \
            f"Need proper ROI shape: {roi_shape}"
        assert isinstance(overlap, float) and overlap > 0 and overlap < 1, \
            f"Need sliding window overlap factor in (0,1): {overlap}"

        print(type(image))

        image_shape = list(image.shape[2:])
        dim = len(image_shape)
        strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]

        size = [(image_shape[i] - roi_shape[i]) //
                strides[i] + 1 for i in range(dim)]

        for i in range(0, strides[0] * size[0], strides[0]):
            for j in range(0, strides[1] * size[1], strides[1]):
                for k in range(0, strides[2] * size[2], strides[2]):
                    yield i, j, k

    def prepare_arrays(self, image, roi_shape=ROI_SHAPE):
        """
        Returns empty arrays required for sliding window inference such as:
        - result array where sub-volume inference results are gathered
        - norm_map where normal map is constructed upon
        - norm_patch, a gaussian kernel that is applied to each sub-volume inference result
        """
        assert isinstance(roi_shape, list) and len(roi_shape) == 3 and any(roi_shape), \
            f"Need proper ROI shape: {roi_shape}"

        image_shape = list(image.shape[2:])
        result = np.zeros(shape=(1, 3, *image_shape), dtype=image.dtype)

        norm_map = np.zeros_like(result)

        # arguments passed are: 128- Number of points in the output window & 16 - the standard deviation
        # a filter to apply for semantic segmentation
        norm_patch = pre_p.gaussian_kernel(
            roi_shape[0], 0.125 * roi_shape[0]).astype(norm_map.dtype)

        return result, norm_map, norm_patch

    def summarize_accuracy(self):
        """
        A function summarizing the accuracy achieved on the images obtained with get_input_array() calls on which
        predictions done where supplied with submit_predictions() function.
        """
        print('to be implemented')
