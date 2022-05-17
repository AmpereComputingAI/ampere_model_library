# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import os
import cv2
import json
import pathlib
import utils.cv.coco as coco_utils
import utils.misc as utils
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def open_json_file(json_file_path):
    with open(json_file_path) as json_file:
        return json.load(json_file)


def convert_fake_bbox(images_path, images_filename_base, image_id, target_shape, bbox):
    image_path = images_filename_base[:-len(str(image_id))] + str(image_id) + ".jpg"
    image_path = pathlib.PurePath(images_path, image_path)
    image_array = cv2.imread(str(image_path))
    vertical_ratio = target_shape[0] / image_array.shape[0]
    horizontal_ratio = target_shape[1] / image_array.shape[1]
    bbox[0] *= horizontal_ratio
    bbox[1] *= vertical_ratio
    bbox[2] *= horizontal_ratio
    bbox[3] *= vertical_ratio
    return bbox


def run_wrapper(test_data_path, num_test_images=100):
    print("\nRUNNING WRAPPER ...\n")
    images_filename_base = "COCO_val2014_000000000000"
    coco = coco_utils.COCODataset(1, images_filename_base, sort_ascending=True)
    env_var = "COCO_IMG_PATH"
    images_path = utils.get_env_variable(
        env_var, f"Path to COCO images directory has not been specified with {env_var} flag")
    test_data = open_json_file(test_data_path)

    some_shape = (300, 300)  # doesn't matter what shape you will set here (but don't use zeros!!!)

    id = -1
    iter = 0
    for test_bbox in test_data:
        if test_bbox["image_id"] > id:
            _ = coco.get_input_array(some_shape)
            if test_bbox["image_id"] == 1064:
                # there is one (id=1063) image in val 2014 dataset
                # that is skipped in fake bbox file from COCO Api for some reason
                _ = coco.get_input_array(some_shape)
            id = test_bbox["image_id"]
            if iter == num_test_images:
                break
            iter += 1
        coco.submit_bbox_prediction(
            0,
            convert_fake_bbox(images_path, images_filename_base, id, some_shape, test_bbox["bbox"]),
            test_bbox["score"],
            test_bbox["category_id"]
        )

    coco.summarize_accuracy()


def run_reference(test_data_path, num_test_images=100):
    print("\nRUNNING REFERENCE ...\n")
    env_var = "COCO_ANNO_PATH"
    annotations_path = utils.get_env_variable(
        env_var, f"Path to COCO annotations has not been specified with {env_var} flag")
    coco_gt = COCO(annotations_path)
    coco_dt = coco_gt.loadRes(test_data_path)
    img_ids = sorted(coco_gt.getImgIds())
    img_ids = img_ids[0:num_test_images]
    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.imgIds = img_ids
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()


def main():
    env_var = "COCO_TEST_PREDS"
    test_data_path = utils.get_env_variable(
        env_var, f"Path to COCO annotations has not been specified with {env_var} flag")
    run_reference(test_data_path)
    run_wrapper(test_data_path)


if __name__ == "__main__":
    main()
