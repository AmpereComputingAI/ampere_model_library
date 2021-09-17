import os
import argparse
# from utils.tf import TFSavedModelRunner
from pathlib import Path
import pickle
import tensorflow as tf
import numpy as np
from scipy import signal

DATASET = '/onspecta/dev/mz/temp/datasets/kits19_preprocessed/preprocessed_files.pkl'
DATASET_DIR = '/onspecta/dev/mz/temp/datasets/kits19_preprocessed'
MODEL_PATH = '/onspecta/dev/mz/temp/models/unet'

DATASET_GRAVITON = '/onspecta/mz/temp/datasets/kits19_preprocessed/preprocessed_files.pkl'
DATASET_DIR_GRAVITON = '/onspecta/mz/temp/datasets/kits19_preprocessed'
MODEL_PATH_GRAVITON = '/onspecta/mz/temp/models/unet'

SAMPLE_LIST = [14, 32, 33, 23, 25, 31, 0, 5, 39, 21, 9, 19, 29, 38, 20, 30]
ROI_SHAPE = [128, 128, 128]
SLIDE_OVERLAP_FACTOR = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="Run Unet model.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=False,
                        help="path to the model")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--images_path",
                        type=str,
                        help="path to directory with ImageNet validation images")
    parser.add_argument("--labels_path",
                        type=str,
                        help="path to file with validation labels")
    return parser.parse_args()


def run_tf_fp32(model_path, batch_size, num_of_runs, timeout, images_path, labels_path):
    # AN ARRAY OF LOADED FILES
    loaded_files = {}

    # LOADING A MODEL
    loaded_model = tf.saved_model.load(MODEL_PATH_GRAVITON)
    model = loaded_model.signatures["serving_default"]

    # GET THE LIST OF PREPROCESSED FILES
    with open(Path(DATASET_GRAVITON), "rb") as f:
        preprocess_files = pickle.load(f)['file_list']

    # GET THE FILE
    file_name = preprocess_files[14]
    with open(Path(DATASET_DIR_GRAVITON, "{:}.pkl".format(file_name)), "rb") as f:
        loaded_files[14] = pickle.load(f)[0]

    # EXPAND ARRAY
    image = loaded_files[14][np.newaxis, ...]
    print(image.shape)

    result, norm_map, norm_patch = prepare_arrays(image, ROI_SHAPE)

    # t_image, t_result, t_norm_map, t_norm_patch = to_tensor(image), \
    #                                               to_tensor(result), \
    #                                               to_tensor(norm_map), \
    #                                               to_tensor(norm_patch)

    # sliding window inference
    subvol_cnt = 0
    for i, j, k in get_slice_for_sliding_window(image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
        subvol_cnt += 1
        result_slice = result[
                       ...,
                       i:(ROI_SHAPE[0] + i),
                       j:(ROI_SHAPE[1] + j),
                       k:(ROI_SHAPE[2] + k)]

        input_slice = image[
                      ...,
                      i:(ROI_SHAPE[0] + i),
                      j:(ROI_SHAPE[1] + j),
                      k:(ROI_SHAPE[2] + k)]

        norm_map_slice = norm_map[
                         ...,
                         i:(ROI_SHAPE[0] + i),
                         j:(ROI_SHAPE[1] + j),
                         k:(ROI_SHAPE[2] + k)]

        output_name = list(model.structured_outputs)[0]
        result_slice += model(tf.constant(input_slice))[output_name].numpy() * norm_patch

        norm_map_slice += norm_patch

    # result, norm_map = from_tensor(result), from_tensor(norm_map)

    final_result = finalize(result, norm_map)

    print(final_result)

    print('done')


def prepare_arrays(image, roi_shape=ROI_SHAPE):
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
    norm_patch = gaussian_kernel(
        roi_shape[0], 0.125 * roi_shape[0]).astype(norm_map.dtype)

    return result, norm_map, norm_patch


def gaussian_kernel(n, std):
    """
    Returns gaussian kernel; std is standard deviation and n is number of points
    Gaussian blur
    return: a numpy array,
    """
    gaussian1d = signal.gaussian(n, std)
    gaussian2d = np.outer(gaussian1d, gaussian1d)
    gaussian3d = np.outer(gaussian2d, gaussian1d)
    gaussian3d = gaussian3d.reshape(n, n, n)
    gaussian3d = np.cbrt(gaussian3d)
    gaussian3d /= gaussian3d.max()

    return gaussian3d


def finalize(image, norm_map):
    """
    Finalizes results obtained from sliding window inference
    """
    # NOTE: layout is assumed to be linear (NCDHW) always
    # apply norm_map
    image = apply_norm_map(image, norm_map)

    # argmax
    image = apply_argmax(image)

    return image


def apply_norm_map(image, norm_map):
    """
    Applies normal map norm_map to image and return the outcome
    """
    image /= norm_map
    return image


def apply_argmax(image):
    """
    Returns indices of the maximum values along the channel axis
    Input shape is (bs=1, channel=3, (ROI_SHAPE)), float -- sub-volume inference result
    Output shape is (bs=1, channel=1, (ROI_SHAPE)), integer -- segmentation result
    """
    channel_axis = 1
    image = np.argmax(image, axis=channel_axis).astype(np.uint8)
    image = np.expand_dims(image, axis=0)

    return image


def to_tensor(my_array):
    """
    Transforms my_array into tensor form backend understands
    Implementation may differ as backend's need
    """
    return my_array


def from_tensor(my_tensor):
    """
    Transforms my_tensor backend worked on into numpy friendly array
    Implementation may differ as backend's need
    """
    return my_tensor


def get_slice_for_sliding_window(image, roi_shape=ROI_SHAPE, overlap=SLIDE_OVERLAP_FACTOR):
    """
    Returns indices for image stride, to fulfill sliding window inference
    Stride is determined by roi_shape and overlap
    """
    assert isinstance(roi_shape, list) and len(roi_shape) == 3 and any(roi_shape), \
        f"Need proper ROI shape: {roi_shape}"
    assert isinstance(overlap, float) and overlap > 0 and overlap < 1, \
        f"Need sliding window overlap factor in (0,1): {overlap}"

    image_shape = list(image.shape[2:])
    dim = len(image_shape)
    strides = [int(roi_shape[i] * (1 - overlap)) for i in range(dim)]

    size = [(image_shape[i] - roi_shape[i]) //
            strides[i] + 1 for i in range(dim)]

    for i in range(0, strides[0] * size[0], strides[0]):
        for j in range(0, strides[1] * size[1], strides[1]):
            for k in range(0, strides[2] * size[2], strides[2]):
                yield i, j, k


def do_infer(self, input_tensor):
    """
    Perform inference upon input_tensor with TensorFlow
    """
    return model(tf.constant(input_tensor))[self.output_name].numpy()


def main():
    args = parse_args()
    run_tf_fp32(
        args.model_path, 1, args.num_runs, args.timeout, args.images_path, args.labels_path
    )


if __name__ == "__main__":
    main()
