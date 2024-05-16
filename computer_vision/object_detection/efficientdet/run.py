# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
try:
    from utils import misc  # noqa
except ModuleNotFoundError:
    import os
    import sys
    filename = "set_env_variables.sh"
    directory = os.path.realpath(__file__).split("/")[:-1]
    for idx in range(1, len(directory) - 1):
        subdir = "/".join(directory[:-idx])
        if filename in os.listdir(subdir):
            print(f"\nPlease run \033[91m'source {os.path.join(subdir, filename)}'\033[0m first.")
            break
    else:
        print(f"\n\033[91mFAIL: Couldn't find {filename}, are you running this script as part of Ampere Model Library?"
              f"\033[0m")
    sys.exit(1)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run EfficientDet model.")
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["int8"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str,
                        choices=["tf"], required=True,
                        help="specify the framework in which a model should be run")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--images_path",
                        type=str,
                        help="path to directory with COCO validation images")
    parser.add_argument("--anno_path",
                        type=str,
                        help="path to file with validation annotations")
    return parser.parse_args()


def run_tflite(model_path, batch_size, num_runs, timeout, images_path, anno_path):
    import numpy as np
    from utils.cv.coco import COCODataset
    from utils.benchmark import run_model
    from utils.tflite import TFLiteRunner

    def run_single_pass(tflite_runner, coco):
        shape = (448, 448)
        tflite_runner.set_input_tensor(
            tflite_runner.input_details[0]["index"], coco.get_input_array(shape).astype(np.uint8))
        tflite_runner.run(batch_size)
        detection_boxes = tflite_runner.get_output_tensor(tflite_runner.output_details[0]["index"])
        detection_classes = tflite_runner.get_output_tensor(tflite_runner.output_details[1]["index"])
        detection_classes += 1  # model uses indexing from 0 while COCO dateset start with idx of 1
        detection_scores = tflite_runner.get_output_tensor(tflite_runner.output_details[2]["index"])
        num_detections = tflite_runner.get_output_tensor(tflite_runner.output_details[3]["index"])

        for i in range(batch_size):
            for d in range(int(num_detections[i])):
                coco.submit_bbox_prediction(
                    i,
                    coco.convert_bbox_to_coco_order(detection_boxes[i][d] * shape[0], 1, 0, 3, 2),
                    detection_scores[i][d],
                    int(detection_classes[i][d])
                )

    dataset = COCODataset(batch_size, "RGB", "COCO_val2014_000000000000", images_path, anno_path,
                          pre_processing=None, sort_ascending=True)
    runner = TFLiteRunner(model_path)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_tflite_int8(model_path, batch_size, num_runs, timeout, images_path, anno_path, **kwargs):
    return run_tflite(model_path, batch_size, num_runs, timeout, images_path, anno_path)


def main():
    from utils.misc import print_goodbye_message_and_die
    args = parse_args()
    if args.framework == "tf":
        if args.model_path is None:
            print_goodbye_message_and_die(
                "a path to model is unspecified!")
        if args.precision == "int8":
            run_tflite_int8(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)
    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
