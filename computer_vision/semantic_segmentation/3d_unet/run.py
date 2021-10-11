import argparse

from utils.kits import KiTS19
from utils.runners import UnetRunner
from tensorflow.python.saved_model import tag_constants
from utils.tf import TFSavedModelRunner
from utils.benchmark import run_model
from utils.global_vars import ROI_SHAPE, SLIDE_OVERLAP_FACTOR
import numpy as np
import time
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="Run 3D Unet model.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=True,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--images_path",
                        type=str, required=True,
                        help="path to directory with KiTS19 dataset")
    parser.add_argument("--anno_path",
                        type=str, required=True,
                        help="path to pickle file containing KiTS19 dataset case names."
                             "its default name is preprocessed_files.pkl")
    parser.add_argument("--groundtruth_path",
                        type=str, required=True,
                        help="path to nifti folder in preprocessed kits directory")
    return parser.parse_args()


def run_tf_fp32(model_path, batch_size, num_of_runs, timeout, images_path, anno_path, groundtruth_path):
    def run_single_pass(tf_runner, kits):
        #shape = (416, 416)
        image, result, norm_map, norm_patch = kits.get_input_array()
        image = np.random.rand(1, 1, 128, 256, 256).astype("float32")
        z = tf.constant(image)
        start = time.time()
        x = tf_runner.run(z)
        ende = time.time()
        print(ende - start)
        #kits.submit_predictions(tf_runner.run(tf.constant(image)))

    # dataset = COCODataset(batch_size, "RGB", "COCO_val2014_000000000000", images_path, anno_path,
    #                       pre_processing="YOLO", sort_ascending=True)
    dataset = KiTS19(images_path=images_path, images_anno=anno_path, groundtruth_path=groundtruth_path)
    runner = TFSavedModelRunner()
    saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
    runner.model = saved_model_loaded.signatures['serving_default']

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def run_tf_fp32_org(model_path, num_of_runs, timeout, images_path, anno_path, groundtruth_path):

    def run_single_pass(unet_runner, kits_dataset):

        image, result, norm_map, norm_patch = kits_dataset.get_input_array()
        print(image.shape)


        for i, j, k in kits_dataset.get_slice_for_sliding_window(image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):

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
            print(ROI_SHAPE)
            print(i, j, k)
            print(input_slice.shape)

            norm_map_slice = norm_map[
                             ...,
                             i:(ROI_SHAPE[0] + i),
                             j:(ROI_SHAPE[1] + j),
                             k:(ROI_SHAPE[2] + k)]

            output = unet_runner.run(tf.constant(input_slice))
            result_slice += output[unet_runner.output_name].numpy() * norm_patch
            #norm_map_slice += norm_patch

        final_result = kits_dataset.finalize(result, norm_map)[0, 0, :, :, :]
        kits_dataset.submit_predictions(final_result)

    dataset = KiTS19(images_path=images_path, images_anno=anno_path, groundtruth_path=groundtruth_path)
    runner = UnetRunner(model_path)

    return run_model(run_single_pass, runner, dataset, 1, num_of_runs, timeout)


def main():
    args = parse_args()
    run_tf_fp32(
        args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.anno_path, args.groundtruth_path
    )


if __name__ == "__main__":
    main()
