import os
import time
import argparse
import utils.misc as utils
from utils.coco import COCODataset
from utils.tflite import TFLiteRunner
from utils.tf import TFSavedModelRunner
from utils.benchmark import run_model
import tensorflow as tf


COCO_MAP = {1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: 9, 10: 10, 11: 11, 12: 13, 13: 14, 14: 15, 15: 16,
            16: 17, 17: 18, 18: 19, 19: 20, 20: 21, 21: 22, 22: 23, 23: 24, 24: 25, 25: 27, 26: 28, 27: 31, 28: 32,
            29: 33, 30: 34, 31: 35, 32: 36, 33: 37, 34: 38, 35: 39, 36: 40, 37: 41, 38: 42, 39: 43, 40: 44, 41: 46,
            42: 47, 43: 48, 44: 49, 45: 50, 46: 51, 47: 52, 48: 53, 49: 54, 50: 55, 51: 56, 52: 57, 53: 58, 54: 59,
            55: 60, 56: 61, 57: 62, 58: 63, 59: 64, 60: 65, 61: 67, 62: 70, 63: 72, 64: 73, 65: 74, 66: 75, 67: 76,
            68: 77, 69: 78, 70: 79, 71: 80, 72: 81, 73: 82, 74: 84, 75: 85, 76: 86, 77: 87, 78: 88, 79: 89, 80: 90}


def parse_args():
    parser = argparse.ArgumentParser(description="Run SSD MobileNet v2 model.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=True,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32", "int8"], required=True,
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
    def run_single_pass(tf_runner, coco):
        shape = (416, 416)
        output = tf_runner.run(coco.get_input_array(shape))
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
                    COCO_MAP[int(detection_classes[i][d]+1)]
                )

    dataset = COCODataset(batch_size, "RGB", "COCO_val2014_000000000000", images_path, anno_path,
                          pre_processing="YOLO", sort_ascending=True)
    runner = TFSavedModelRunner(model_path)

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def run_tflite_int8(model_path, batch_size, num_of_runs, timeout, images_path, anno_path):
    def run_single_pass(tflite_runner, coco):
        shape = (416, 416)
        tflite_runner.set_input_tensor(tflite_runner.input_details[0]["index"], coco.get_input_array(shape))
        tflite_runner.run()
        detection_boxes = tflite_runner.get_output_tensor(tflite_runner.output_details[0]["index"])
        print(detection_boxes.shape)
        #for
        # detection_classes = tflite_runner.get_output_tensor(tflite_runner.output_details[1]["index"])
        # detection_classes += 1  # model uses indexing from 0 while COCO dateset start with idx of 1
        # detection_scores = tflite_runner.get_output_tensor(tflite_runner.output_details[2]["index"])
        # num_detections = tflite_runner.get_output_tensor(tflite_runner.output_details[3]["index"])
        # for i in range(batch_size):
        #     for d in range(int(num_detections[i])):
        #         coco.submit_bbox_prediction(
        #             i,
        #             coco.convert_bbox_to_coco_order(detection_boxes[i][d] * shape[0], 1, 0, 3, 2),
        #             detection_scores[i][d],
        #             int(detection_classes[i][d])
        #         )

    dataset = COCODataset(batch_size, "RGB", "COCO_val2014_000000000000", images_path, anno_path,
                          pre_processing="SSD", sort_ascending=True)
    runner = TFLiteRunner(model_path)

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def main():
    args = parse_args()
    if args.precision == "fp32":
        run_tf_fp32(
            args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.anno_path
        )
    elif args.precision == "int8":
        run_tflite_int8(
            args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.anno_path
        )
    else:
        assert False


if __name__ == "__main__":
    main()
