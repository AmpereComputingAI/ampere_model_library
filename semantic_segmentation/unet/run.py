import argparse
from utils.kits import Kits
from utils.tf import TFSavedModelRunner
from utils.benchmark import run_model
from utils.global_vars import ROI_SHAPE, SLIDE_OVERLAP_FACTOR
import tensorflow as tf

DATASET = '/onspecta/dev/mz/temp/datasets/kits19_preprocessed/preprocessed_files.pkl'
DATASET_DIR = '/onspecta/dev/mz/temp/datasets/kits19_preprocessed'
MODEL_PATH = '/onspecta/dev/mz/temp/models/unet'

DATASET_GRAVITON = '/onspecta/mz/temp/datasets/kits19_preprocessed/preprocessed_files.pkl'
DATASET_DIR_GRAVITON = '/onspecta/mz/temp/datasets/kits19_preprocessed'
MODEL_PATH_GRAVITON = '/onspecta/mz/temp/models/unet'


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


def run_tf_fp32(model_path, num_of_runs, timeout, images_path):

    def run_single_pass(unet_runner, kits_dataset):

        image, result, norm_map, norm_patch = kits_dataset.get_input_array()

        subvol_cnt = 0
        for i, j, k in kits_dataset.get_slice_for_sliding_window(image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
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

            output = unet_runner.run(tf.constant(input_slice))
            print('here')
            result_slice += output[unet_runner.output_name].numpy() * norm_patch
            norm_map_slice += norm_patch

    dataset = Kits(images_path=DATASET_DIR_GRAVITON, images_anno=DATASET_GRAVITON)
    runner = TFSavedModelRunner(MODEL_PATH_GRAVITON)

    return run_model(run_single_pass, runner, dataset, 1, num_of_runs, timeout)


def main():
    args = parse_args()
    run_tf_fp32(
        args.model_path, args.num_runs, args.timeout, args.images_path
    )


if __name__ == "__main__":
    main()
