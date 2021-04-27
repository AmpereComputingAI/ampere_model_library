import argparse
from utils.tf import TFFrozenModelRunner
from utils.coco import COCODataset
from utils.benchmark import run_model
from utils.misc import COCO_MAP

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Run MobileNet v2 model.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=True,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32", "fp16", "int8"], required=True,
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
                        type=str,
                        help="path to directory with ImageNet validation images")
    parser.add_argument("--labels_path",
                        type=str,
                        help="path to file with validation labels")
    return parser.parse_args()


def run_tf_fp32(model_path, batch_size, num_of_runs, timeout, images_path, labels_path):
    def run_single_pass(tf_runner, coco):
        shape = (416, 416)
        tf_runner.set_input_tensor("inputs:0", coco.get_input_array(shape))
        output = tf_runner.run()
        output_boxes = output["output_boxes:0"]

        for i in range(batch_size):
            for j in range(output_boxes.shape[1]):

                left_boundary = output_boxes[i][j][0]
                top_boundary = output_boxes[i][j][1]
                right_boundary = output_boxes[i][j][2]
                bottom_boundary = output_boxes[i][j][3]
                confidence_score = output_boxes[i][j][4]
                box = [left_boundary, top_boundary, right_boundary, bottom_boundary]

                classes = output_boxes[i][j][5:]
                coco_index = COCO_MAP[np.argmax(classes) + 1]
                coco.submit_bbox_prediction(
                    i,
                    coco.convert_bbox_to_coco_order(box, 0, 1, 2, 3, True),
                    confidence_score,
                    coco_index
                )

    dataset = COCODataset(batch_size, "RGB", "COCO_val2014_000000000000", images_path, labels_path, sort_ascending=True)

    runner = TFFrozenModelRunner(model_path, ["output_boxes:0"])

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def main():
    args = parse_args()
    if args.precision == "fp32":
        run_tf_fp32(
            args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.labels_path
        )
    else:
        assert False


if __name__ == "__main__":
    main()
