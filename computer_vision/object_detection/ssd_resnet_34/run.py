# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import os
import time
import argparse

import utils.misc as utils
from utils.cv.coco import COCODataset
from utils.benchmark import run_model
from utils.misc import print_goodbye_message_and_die


def parse_args():
    parser = argparse.ArgumentParser(description="Run SSD ResNet-34  model.")
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32", "fp16"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
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
    from utils.tf import TFFrozenModelRunner

    def run_single_pass(tf_runner, coco):
        shape = (1200, 1200)
        tf_runner.set_input_tensor("image:0", coco.get_input_array(shape))
        output = tf_runner.run()
        for i in range(batch_size):
            num_detections = int(output["detection_bboxes:0"][i].shape[0])
            for d in range(num_detections):
                coco.submit_bbox_prediction(
                    i,
                    coco.convert_bbox_to_coco_order(output["detection_bboxes:0"][i][d] * shape[0], 1, 0, 3, 2),
                    output["detection_scores:0"][i][d],
                    coco.translate_cat_id_to_coco(int(output["detection_classes:0"][i][d]),
                                                  switch_to_indexing_from_1=False)
                )
                if output["detection_scores:0"][i][d] < 0.05:
                    break

    dataset = COCODataset(batch_size, "RGB", "COCO_val2014_000000000000", images_path, anno_path,
                          pre_processing="SSD_2", sort_ascending=True)
    runner = TFFrozenModelRunner(
        model_path, ["detection_classes:0", "detection_bboxes:0", "detection_scores:0"])

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_tf_fp32(model_path, batch_size, num_runs, timeout, images_path, anno_path, **kwargs):
    return run_tf_fp(model_path, batch_size, num_runs, timeout, images_path, anno_path)


def run_tf_fp16(model_path, batch_size, num_runs, timeout, images_path, anno_path, **kwargs):
    return run_tf_fp(model_path, batch_size, num_runs, timeout, images_path, anno_path)


def main():
    args = parse_args()
    if args.framework == "tf":
        if args.model_path is None:
            print_goodbye_message_and_die(
                "a path to model is unspecified!")

        if args.precision == "fp32":
            run_tf_fp32(**vars(args))
        elif args.precision == "fp16":
            run_tf_fp16(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
