import os
import time
import argparse
import utils.tf as tf_utils
import utils.coco as coco_utils
import tensorflow.compat.v1 as tf


def parse_args():
    parser = argparse.ArgumentParser(description="Run SSD MobileNet v2 model.")
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
                        help="path to directory with COCO validation images")
    parser.add_argument("--anno_path",
                        type=str,
                        help="path to file with validation annotations")
    return parser.parse_args()


def run_tf_fp32(model_path, batch_size, num_of_runs, timeout, images_path, anno_path):
    shape = (640, 640)
    coco = coco_utils.COCODataset(batch_size, "COCO_val2014_000000000000", images_path, anno_path)

    runner = tf_utils.TFFrozenModelRunner(
        model_path,
        ["detection_classes:0", "detection_boxes:0", "detection_scores:0", "num_detections:0"]
    )

    iter = 0
    start = time.time()
    while True:
        if num_of_runs is None:
            if time.time() - start > timeout:
                break
        elif not iter < num_of_runs:
            break

        try:
            runner.set_input_tensor("image_tensor:0", coco.get_input_array(shape))
        except coco.OutOfCOCOImages:
            break

        output = runner.run()
        iter += 1

        for i in range(batch_size):
            for d in range(int(output["num_detections:0"][i])):
                coco.submit_bbox_prediction(
                    i,
                    coco.convert_bbox_to_coco_order(output["detection_boxes:0"][i][d] * shape[0], 1, 0, 3, 2),
                    output["detection_scores:0"][i][d],
                    int(output["detection_classes:0"][i][d])
                )

    coco.summarize_accuracy()
    runner.print_performance_metrics(batch_size)


def main():
    args = parse_args()
    if args.precision == "fp32":
        run_tf_fp32(
            args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.anno_path
        )
    else:
        assert False, "Other precisions not supported yet."


if __name__ == "__main__":
    main()
