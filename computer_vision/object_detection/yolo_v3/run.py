# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import argparse

import numpy as np

from utils.cv.coco import COCODataset
from utils.benchmark import run_model
from utils.misc import print_goodbye_message_and_die


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLO v3 model.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=True,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str, default="ort",
                        choices=["ort"],
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


def run_ort_fp32(model_path, batch_size, num_runs, timeout, images_path, anno_path, **kwargs):
    from utils.ort import OrtRunner

    def run_single_pass(ort_runner, coco):
        shape = (416, 416)
        ort_runner.set_input_tensor("input_1", coco.get_input_array(shape).astype("float32"))
        ort_runner.set_input_tensor("image_shape", np.array(shape, dtype="float32").reshape(1, 2))
        output = ort_runner.run(batch_size)

        boxes = output[0]
        scores = output[1]
        indices = output[2]

        out_boxes, out_scores, out_classes = [], [], []
        for idx_ in indices:
            out_classes.append(idx_[1])
            out_scores.append(scores[tuple(idx_)])
            idx_1 = (idx_[0], idx_[2])
            out_boxes.append(boxes[idx_1])

        for d, box in enumerate(out_boxes):
            coco.submit_bbox_prediction(
                0,
                coco.convert_bbox_to_coco_order(box, 1, 0, 3, 2),
                out_scores[d],
                coco.translate_cat_id_to_coco(int(out_classes[d]))
            )

    dataset = COCODataset(batch_size, "RGB", "COCO_val2014_000000000000", images_path, anno_path,
                          pre_processing="YOLO", order="NCHW")
    runner = OrtRunner(model_path)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def main():
    args = parse_args()
    if args.framework == "ort":
        if args.precision == "fp32":
            if args.batch_size != 1:
                raise ValueError("Batch size must be 1 for this model.")
            run_ort_fp32(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
