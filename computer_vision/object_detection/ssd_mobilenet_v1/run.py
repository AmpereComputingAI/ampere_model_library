import os
import time
import argparse
from utils.cv.coco import COCODataset
from utils.tf import TFFrozenModelRunner
from utils.benchmark import run_model
from utils.misc import UnsupportedPrecisionValueError, FrameworkUnsupportedError


def parse_args():
    parser = argparse.ArgumentParser(description="Run SSD MobileNet v1 model.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=True,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32", "fp16"], required=True,
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
    parser.add_argument("--framework",
                        type=str,
                        choices=["tf"], required=True,
                        help="specify the framework in which a model should be run")
    return parser.parse_args()


def run_tf_fp(model_path, batch_size, num_of_runs, timeout, images_path, anno_path):
    def run_single_pass(tf_runner, coco):
        shape = (300, 300)
        tf_runner.set_input_tensor("image_tensor:0", coco.get_input_array(shape))
        output = tf_runner.run()
        for i in range(batch_size):
            for d in range(int(output["num_detections:0"][i])):
                coco.submit_bbox_prediction(
                    i,
                    coco.convert_bbox_to_coco_order(output["detection_boxes:0"][i][d] * shape[0], 1, 0, 3, 2),
                    output["detection_scores:0"][i][d],
                    int(output["detection_classes:0"][i][d])
                )

    dataset = COCODataset(batch_size, "RGB", "COCO_val2014_000000000000", images_path, anno_path, sort_ascending=True)
    runner = TFFrozenModelRunner(
        model_path, ["detection_classes:0", "detection_boxes:0", "detection_scores:0", "num_detections:0"])

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def run_tf_fp32(model_path, batch_size, num_of_runs, timeout, images_path, anno_path):
    return run_tf_fp(model_path, batch_size, num_of_runs, timeout, images_path, anno_path)


def run_tf_fp16(model_path, batch_size, num_of_runs, timeout, images_path, anno_path):
    return run_tf_fp(model_path, batch_size, num_of_runs, timeout, images_path, anno_path)


def main():
    args = parse_args()
    if args.framework == "tf":
        if args.precision == "fp32":
            run_tf_fp32(
                args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.anno_path
            )
        elif args.precision == "fp16":
            run_tf_fp16(
                args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.anno_path
            )
        else:
            raise UnsupportedPrecisionValueError(args.precision)
    else:
        raise FrameworkUnsupportedError(args.framework)


if __name__ == "__main__":
    main()
