import os
import time
import argparse
import utils.misc as utils
from utils.coco import COCODataset
from utils.tflite import TFLiteRunner
from utils.tf import TFFrozenModelRunner
from utils.benchmark import run_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run SSD ResNet-34  model.")
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
                        type=str,
                        help="path to directory with COCO validation images")
    parser.add_argument("--anno_path",
                        type=str,
                        help="path to file with validation annotations")
    return parser.parse_args()


def run_tf_fp(model_path, batch_size, num_of_runs, timeout, images_path, anno_path):
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
                    coco.translate_cat_id_to_coco(int(output["detection_classes:0"][i][d],
                                                      switch_to_indexing_from_1=False))
                )
                if output["detection_scores:0"][i][d] < 0.05:
                    break

    dataset = COCODataset(batch_size, "RGB", "COCO_val2014_000000000000", images_path, anno_path,
                          pre_processing="SSD_2", sort_ascending=True)
    runner = TFFrozenModelRunner(
        model_path, ["detection_classes:0", "detection_bboxes:0", "detection_scores:0"])

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def run_tf_fp32(model_path, batch_size, num_of_runs, timeout, images_path, anno_path):
    return run_tf_fp(model_path, batch_size, num_of_runs, timeout, images_path, anno_path)


def main():
    args = parse_args()
    if args.precision == "fp32":
        run_tf_fp32(
            args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.anno_path
        )
    else:
        assert False


if __name__ == "__main__":
    main()
