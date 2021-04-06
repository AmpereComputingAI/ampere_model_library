import os
import json
import pathlib
import utils.coco_utils as coco_utils


def open_json_file(json_file_path):
    with open(json_file_path) as json_file:
        return json.load(json_file)


def run_coco_test():
    coco = coco_utils.COCODataset(1,
                                  processed_annotations_path=pathlib.PurePath(
                                      os.path.dirname(os.path.realpath(__file__)), "test_annotations.json"),
                                  images_filename_base="COCO_val2014_000000000000")
    print(pathlib.PurePath(os.path.realpath(__file__), "instances_val2014_fakebbox100_results.json"))
    fake_data = open_json_file(
        pathlib.PurePath(os.path.dirname(os.path.realpath(__file__)), "instances_val2014_fakebbox100_results.json"))
    fake_img_id = 0
    iter = 0
    images_shapes = [(478, 640), (640, 565), (426, 640), (480, 640), (374, 500), (426, 640), (500, 600), (480, 640), (480, 640), (480, 640)]
    for fake_img in fake_data:
        if fake_img["image_id"] > fake_img_id:
            if iter > 9:
                break
            fake_img_id = fake_img["image_id"]
            print("yo")
            _ = coco.get_input_array(images_shapes[iter])
            iter += 1

        fake_bbox = fake_img["bbox"]
        fake_bbox[2] += fake_bbox[0]
        fake_bbox[3] += fake_bbox[1]
        coco.submit_bbox_prediction(
            0,
            fake_bbox,
            fake_img["category_id"]
        )
    coco.summarize_accuracy()


if __name__ == "__main__":
    # real_data = open_json_file("/home/jan/onspecta/cocoapi/annotations/instances_val2014.json")
    # for i in real_data["annotations"]:
    #    print(i)
    run_coco_test()
