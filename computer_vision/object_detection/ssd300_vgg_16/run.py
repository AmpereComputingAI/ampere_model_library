import os
import time
import argparse
import utils.misc as utils
import utils.cv.post_processing as pp

import torch
import torchvision

from utils.cv.coco import COCODataset
from utils.pytorch import PyTorchRunner
from utils.benchmark import run_model

import csv

from utils.misc import UnsupportedPrecisionValueError, FrameworkUnsupportedError

IMAGES_PATH = '/onspecta/dev/mz/temp/datasets/COCO2014_onspecta'
ANNO_PATH = '/onspecta/dev/mz/temp/labels/COCO2014_anno_onspecta.json'


def parse_args():
    parser = argparse.ArgumentParser(description="Run SSD Inception v2 model.")
    parser.add_argument("-m", "--model_path",
                        type=str,
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
                        choices=["pytorch"], required=True,
                        help="specify the framework in which a model should be run")
    return parser.parse_args()


def run_pytorch_fp(batch_size, num_of_runs, timeout, images_path, anno_path, iou_tres, score_tres):
    print(iou_tres, score_tres)

    def run_single_pass(pytorch_runner, coco):
        shape = (300, 300)
        output = pytorch_runner.run(coco.get_input_array(shape))

        # ========================================================================================
        # jeden wariant \/ z ręcznym ustawianiem iou_tres, oraz score_tres
        # doubled_boxes_removed = torchvision.ops.batched_nms(output[1][0]['boxes'], output[1][0]['scores'],
        #                                                     output[1][0]['labels'], iou_tres)
        #
        # for i in range(batch_size):
        #     for d in doubled_boxes_removed:
        #         if output[1][i]['scores'][d.item()].item() > score_tres:
        #             coco.submit_bbox_prediction(
        #                 i,
        #                 output[1][i]['boxes'][d.item()].tolist(),
        #                 output[1][i]['scores'][d.item()].item(),
        #                 output[1][i]['labels'][d.item()].item()
        #             )

        # print(output[1][0]['scores'][0].item())

        for i in range(batch_size):
            for d in range(output[1][i]['boxes'].shape[0]):
                if output[1][i]['scores'][d].item() > score_tres:
                    coco.submit_bbox_prediction(
                        i,
                        output[1][i]['boxes'][d].tolist(),
                        output[1][i]['scores'][d].item(),
                        output[1][i]['labels'][d].item()
                    )

        # ========================================================================================
        # drugi wariant
        # print(output)
        # quit()
        # for i in range(batch_size):
        #     for d in range(output[1][i]['boxes'].shape[0]):
        #         coco.submit_bbox_prediction(
        #             i,
        #             output[1][i]['boxes'][d].tolist(),
        #             output[1][i]['scores'][d].item(),
        #             output[1][i]['labels'][d].item()
        #         )

    dataset = COCODataset(batch_size, "BGR", "COCO_val2014_000000000000", images_path, anno_path,
                          pre_processing="PyTorch_objdet", sort_ascending=True, order="CHW")
    runner = PyTorchRunner(torchvision.models.detection.ssd300_vgg16(pretrained=True))

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def run_pytorch_fp32(batch_size, num_of_runs, timeout, images_path, anno_path, iou_tres=None, score_tres=None):
    return run_pytorch_fp(batch_size, num_of_runs, timeout, images_path, anno_path, iou_tres, score_tres)


def run_tf_fp16(model_path, batch_size, num_of_runs, timeout, images_path, anno_path):
    return run_tf_fp(model_path, batch_size, num_of_runs, timeout, images_path, anno_path)


def main():
    args = parse_args()

    # ========================================================================================
    # jeden wariant \/ z ręcznym ustawianiem iou_tres, oraz score_tres
    iou_threshold = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75,
                     0.80, 0.85, 0.90, 0.95, 1.0]
    score_threshold = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70,
                       0.75, 0.80, 0.85, 0.90, 0.95]

    # for i in iou_threshold:
    #     for s in score_threshold:
    #             run_pytorch_fp32(args.batch_size, args.num_runs, args.timeout, IMAGES_PATH, ANNO_PATH, i, s)

    for s in score_threshold:
        run_pytorch_fp32(args.batch_size, args.num_runs, args.timeout, IMAGES_PATH, ANNO_PATH, score_tres=s)

    # ========================================================================================
    # drugi wariant
    # if args.framework == "pytorch":
    #     if args.precision == "fp32":
    #         run_pytorch_fp32(
    #             args.batch_size, args.num_runs, args.timeout, IMAGES_PATH, ANNO_PATH
    #         )
    #     else:
    #         raise UnsupportedPrecisionValueError(args.precision)
    # else:
    #     raise FrameworkUnsupportedError(args.framework)


if __name__ == "__main__":
    main()
