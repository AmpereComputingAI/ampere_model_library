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

    loaded_files = {}

    print('loading a model...')
    loaded_model = tf.saved_model.load(MODEL_PATH)
    model = loaded_model.signatures["serving_default"]

    with open(Path(DATASET), "rb") as f:
        preprocess_files = pickle.load(f)['file_list']

    count = len(preprocess_files)
    print(preprocess_files)

    file_name = preprocess_files[14]
    print("Loading file {:}".format(file_name))
    with open(Path(DATASET_DIR, "{:}.pkl".format(file_name)), "rb") as f:
        loaded_files[14] = pickle.load(f)[0]

    im = loaded_files[14]

    image = im[np.newaxis, ...]

    result, norm_map, norm_patch = prepare_arrays(image, ROI_SHAPE)

    print(result)

    t_image, t_result, t_norm_map, t_norm_patch = to_tensor(image), to_tensor(result), to_tensor(norm_map), to_tensor(norm_patch)

    print('BASE SUT 1')

    # sliding window inference
    subvol_cnt = 0
    for i, j, k in get_slice_for_sliding_window(t_image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
        subvol_cnt += 1
        result_slice = t_result[
                       ...,
                       i:(ROI_SHAPE[0] + i),
                       j:(ROI_SHAPE[1] + j),
                       k:(ROI_SHAPE[2] + k)]

        input_slice = t_image[
                      ...,
                      i:(ROI_SHAPE[0] + i),
                      j:(ROI_SHAPE[1] + j),
                      k:(ROI_SHAPE[2] + k)]

        norm_map_slice = t_norm_map[
                         ...,
                         i:(ROI_SHAPE[0] + i),
                         j:(ROI_SHAPE[1] + j),
                         k:(ROI_SHAPE[2] + k)]

        output_name = list(model.structured_outputs)[0]
        result_slice += model(tf.constant(input_slice))[output_name].numpy() * t_norm_patch

        norm_map_slice += t_norm_patch

    result, norm_map = from_tensor(t_result), from_tensor(t_norm_map)

    final_result = finalize(result, norm_map)


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


def prepare_arrays(image, roi_shape=ROI_SHAPE):
    """
    Returns empty arrays required for sliding window inference such as:
    - result array where sub-volume inference results are gathered
    - norm_map where normal map is constructed upon
    - norm_patch, a gaussian kernel that is applied to each sub-volume inference result
    """
    assert isinstance(roi_shape, list) and len(roi_shape) == 3 and any(roi_shape),\
        f"Need proper ROI shape: {roi_shape}"

    image_shape = list(image.shape[2:])

    result = np.zeros(shape=(1, 3, *image_shape), dtype=image.dtype)
    norm_map = np.zeros_like(result)
    norm_patch = gaussian_kernel(
        roi_shape[0], 0.125*roi_shape[0]).astype(norm_map.dtype)

    return result, norm_map, norm_patch


def to_tensor(my_array):
    """
    Transforms my_array into tensor form backend understands
    Implementation may differ as backend's need
    """
    return my_array


def from_tensor(self, my_tensor):
    """
    Transforms my_tensor backend worked on into numpy friendly array
    Implementation may differ as backend's need
    """
    return my_tensor


def gaussian_kernel(n, std):
    """
    Returns gaussian kernel; std is standard deviation and n is number of points
    """
    gaussian1D = signal.gaussian(n, std)
    gaussian2D = np.outer(gaussian1D, gaussian1D)
    gaussian3D = np.outer(gaussian2D, gaussian1D)
    gaussian3D = gaussian3D.reshape(n, n, n)
    gaussian3D = np.cbrt(gaussian3D)
    gaussian3D /= gaussian3D.max()

    return gaussian3D


def get_slice_for_sliding_window(image, roi_shape=ROI_SHAPE, overlap=SLIDE_OVERLAP_FACTOR):
    """
    Returns indices for image stride, to fulfill sliding window inference
    Stride is determined by roi_shape and overlap
    """
    assert isinstance(roi_shape, list) and len(roi_shape) == 3 and any(roi_shape),\
        f"Need proper ROI shape: {roi_shape}"
    assert isinstance(overlap, float) and overlap > 0 and overlap < 1,\
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
