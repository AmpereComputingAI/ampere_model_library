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
        # for bind in range(cur_batch_size):
        #
        #     for ind_bb in range(num_detections):
        #         record = {}
        #         height, width = image_shape_batch[bind]
        #         ymin = output_dict['detection_bboxes'][bind][ind_bb][0] * height
        #         xmin = output_dict['detection_bboxes'][bind][ind_bb][1] * width
        #         ymax = output_dict['detection_bboxes'][bind][ind_bb][2] * height
        #         xmax = output_dict['detection_bboxes'][bind][ind_bb][3] * width
        #         score = output_dict['detection_scores'][bind][ind_bb]
        #         class_id = int(output_dict['detection_classes'][bind][ind_bb])
        #         record['image_id'] = int(image_name_batch[bind])
        #         record['category_id'] = labelmap[class_id]
        #         record['score'] = score
        #         record['bbox'] = [xmin, ymin, xmax - xmin + 1, ymax - ymin + 1]
        #         if score < coco_constants.SCORE_THRESHOLD:
        #             break
        for i in range(batch_size):
            num_detections = int(output["detection_bboxes:0"][i].shape[0])
            for d in range(num_detections):
                coco.submit_bbox_prediction(
                    i,
                    coco.convert_bbox_to_coco_order(output["detection_bboxes:0"][i][d] * shape[0], 1, 2, 3, 0),
                    output["detection_scores:0"][i][d],
                    int(output["detection_classes:0"][i][d])
                )
                if output["detection_scores:0"][i][d] < 0.05:
                    break

    dataset = COCODataset(batch_size, "BGR", "COCO_val2014_000000000000", images_path, anno_path,
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
