# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import os
import time
import argparse
import numpy as np
import utils.misc as utils
from utils.cv.pose_estimation import PoseEstimationDataset
from utils.benchmark import run_model
from utils.misc import print_goodbye_message_and_die
from pycocotools.coco import COCO
import tensorflow as tf

def parse_args():
    parser = argparse.ArgumentParser(description="Run Movenet model.")
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str,
                        choices=["tf"], required=True,
                        help="specify the framework in which a model should be run")
    parser.add_argument("--timeout",
                        type=float, default=120.0,
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
    from utils.tflite import TFLiteRunner

    def run_single_pass(tflite_runner, coco):
        try:
            images, metadatas = next(coco.dataset)
        except StopIteration:
            raise utils.OutOfInstances("No more MoveNet images to process in the dir provided")
        input_image = tf.cast(images, dtype=tf.float32)

        tflite_runner.set_input_tensor(tflite_runner.input_details[0]["index"], input_image)
        tflite_runner.run(batch_size)
        keypoints_with_scores = tflite_runner.get_output_tensor(tflite_runner.output_details[0]["index"])
        coco.submit_keypoint_prediction(images[0], keypoints_with_scores, metadatas[0])

    runner = TFLiteRunner(model_path)
    image_size = runner.input_details[0]['shape'][1]

    dataset = PoseEstimationDataset(anno_path, images_path, image_size)    

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_tflite_fp32(model_path, batch_size, num_runs, timeout, images_path, anno_path, **kwargs):
    return run_tflite(model_path, batch_size, num_runs, timeout, images_path, anno_path)


def main():
    args = parse_args()
    if args.framework == "tf":
        if args.model_path is None:
            print_goodbye_message_and_die(
                "a path to model is unspecified!")
        if args.batch_size!=1:
            raise ValueError("Batch size must be 1 for this model")
        run_tflite_fp32(**vars(args))       
    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
