# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import os
import time
import argparse

import tensorflow as tf
from tensorflow.python.saved_model import tag_constants

import utils.misc as utils
from utils.cv.coco import COCODataset
from utils.benchmark import run_model
from utils.misc import print_goodbye_message_and_die, download_coco_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO v4 tiny model.")
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("--framework",
                        type=str, default="tf",
                        choices=["tf"],
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


def run_tf_fp(model_path, batch_size, num_runs, timeout, images_path, anno_path):
    from utils.tf import TFSavedModelRunner

    def run_single_pass(tf_runner, coco):
        shape = (416, 416)
        output = tf_runner.run(batch_size, tf.constant(coco.get_input_array(shape)))
        bboxes = output["tf.concat_12"][:, :, 0:4]
        preds = output["tf.concat_12"][:, :, 4:]
        detection_boxes, detection_scores, detection_classes, valid_detections = tf.image.combined_non_max_suppression(
            boxes=tf.reshape(bboxes, (tf.shape(bboxes)[0], -1, 1, 4)),
            scores=tf.reshape(
                preds, (tf.shape(preds)[0], -1, tf.shape(preds)[-1])),
            max_output_size_per_class=50,
            max_total_size=50,
            iou_threshold=0.45,
            score_threshold=0.25
        )
        for i in range(batch_size):
            for d in range(int(valid_detections[i])):
                coco.submit_bbox_prediction(
                    i,
                    coco.convert_bbox_to_coco_order(detection_boxes[i][d] * shape[0], 1, 0, 3, 2),
                    detection_scores[i][d],
                    coco.translate_cat_id_to_coco(int(detection_classes[i][d]))
                )

    dataset = COCODataset(batch_size, "RGB", "COCO_val2014_000000000000", images_path, anno_path,
                          pre_processing="YOLO", sort_ascending=True)
    runner = TFSavedModelRunner()
    saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
    runner.model = saved_model_loaded.signatures['serving_default']

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_tf_fp32(model_path, batch_size, num_runs, timeout, images_path, anno_path, **kwargs):
    return run_tf_fp(model_path, batch_size, num_runs, timeout, images_path, anno_path)


def main():
    args = parse_args()
    download_coco_dataset()

    if args.framework == "tf":
        if args.model_path is None:
            print_goodbye_message_and_die(
                "a path to model is unspecified!")

        if args.precision == "fp32":
            run_tf_fp32(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
