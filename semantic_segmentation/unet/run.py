import argparse
from utils.kits import KiTS19
from utils.runners import UnetRunner
from utils.benchmark import run_model
from utils.global_vars import ROI_SHAPE, SLIDE_OVERLAP_FACTOR
import tensorflow as tf


def parse_args():
    parser = argparse.ArgumentParser(description="Run Unet model.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=True,
                        help="path to the model")
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
    return parser.parse_args()


def run_tf_fp32(model_path, num_of_runs, timeout, images_path, anno_path):

    def run_single_pass(unet_runner, kits_dataset):

        image, result, norm_map, norm_patch = kits_dataset.get_input_array()
        print(result.shape)

        # subvol_cnt = 0
        for i, j, k in kits_dataset.get_slice_for_sliding_window(image, ROI_SHAPE, SLIDE_OVERLAP_FACTOR):
            # subvol_cnt += 1

            # roi = ..., slice(i, ROI_SHAPE[0] + i), ...
            # result_slice = result[[roi]
            # input_slice = image[[roi]
            #     ...
            #
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
            result_slice += output[unet_runner.output_name].numpy() * norm_patch
            norm_map_slice += norm_patch

    dataset = KiTS19(images_path=images_path, images_anno=anno_path)
    runner = UnetRunner(model_path)

    return run_model(run_single_pass, runner, dataset, 1, num_of_runs, timeout)


def main():
    args = parse_args()
    run_tf_fp32(
        args.model_path, args.num_runs, args.timeout, args.images_path, args.anno_path
    )


if __name__ == "__main__":
    main()
