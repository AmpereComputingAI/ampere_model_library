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
    images_shapes = [(478, 640), (640, 565), (426, 640), (480, 640), (374, 500), (426, 640), (500, 600), (480, 640), (480, 640), (480, 640), (480, 640), (640, 480), (480, 640), (640, 428), (640, 586), (427, 640), (491, 640), (327, 640), (218, 640), (332, 500), (375, 500), (480, 640), (580, 640), (480, 640), (640, 638), (640, 359), (360, 640), (640, 516), (226, 640), (500, 333), (427, 640), (406, 640), (427, 640), (480, 640), (336, 448), (427, 640), (640, 520), (480, 640), (480, 640), (500, 409), (407, 640), (500, 375), (480, 640), (483, 640), (640, 480), (428, 640), (375, 500), (480, 640), (480, 640), (428, 640), (267, 400), (480, 640), (480, 640), (500, 375), (428, 640), (427, 640), (480, 640), (343, 500), (424, 640), (427, 640), (640, 428), (425, 640), (640, 424), (480, 640), (496, 500), (500, 333), (335, 500), (640, 621), (640, 502), (427, 640), (485, 640), (480, 640), (640, 427), (640, 425), (500, 334), (426, 640), (480, 640), (347, 640), (480, 640), (640, 443), (640, 480), (335, 500), (427, 640), (573, 640), (444, 640), (375, 500), (640, 427), (427, 640), (375, 500), (427, 640), (426, 640), (640, 426), (640, 426), (480, 640), (480, 640), (479, 640), (427, 640), (480, 640), (540, 640), (427, 640), (640, 427), (375, 500), (480, 640), (375, 500), (500, 375), (480, 640), (425, 640), (640, 437), (482, 407), (640, 478), (512, 640), (375, 500), (480, 640), (375, 500), (427, 640), (315, 640), (240, 320), (371, 640), (480, 640), (427, 640), (480, 640), (640, 435), (480, 640), (640, 424), (612, 612), (640, 480), (640, 640), (640, 425), (480, 640), (375, 500), (304, 480), (428, 640), (640, 424), (480, 640), (480, 640), (427, 640), (480, 640), (428, 640), (453, 640), (640, 523), (640, 427), (375, 500), (480, 640), (500, 500), (425, 640), (427, 640), (363, 500), (640, 427), (427, 640), (424, 640), (375, 500), (640, 383), (640, 427), (640, 427), (375, 500), (419, 640), (480, 640), (427, 640), (270, 360), (480, 640), (427, 640), (388, 500), (427, 640), (480, 640), (480, 640), (427, 640), (425, 640), (427, 640), (480, 640), (323, 640), (426, 640), (375, 500), (427, 640), (424, 640), (600, 600), (427, 640), (427, 640), (640, 480), (457, 640), (302, 500), (480, 640), (480, 640), (480, 640), (375, 500), (640, 457), (480, 640), (360, 640), (427, 640), (364, 500), (612, 612), (375, 500), (427, 640), (500, 375), (640, 480), (427, 640), (375, 500), (366, 640), (427, 640), (375, 500), (480, 640), (480, 640), (555, 640), (427, 640), (427, 640), (426, 640), (479, 640), (480, 640), (480, 640), (428, 640), (484, 640), (491, 640), (640, 427), (375, 500), (480, 640), (507, 640), (375, 500), (480, 640), (500, 375), (425, 640), (480, 640), (480, 640), (480, 640), (480, 640), (338, 400), (640, 480), (364, 640), (428, 640), (427, 640), (640, 464), (640, 443), (480, 640), (640, 640), (640, 480), (338, 640), (363, 640), (640, 480), (480, 640), (640, 480), (427, 640), (395, 640), (640, 427), (390, 640), (427, 640), (375, 500), (427, 640), (333, 500), (612, 612), (427, 640), (480, 640), (480, 640), (640, 491), (384, 640), (480, 640), (428, 640), (211, 640), (427, 640), (640, 480), (480, 640), (480, 640), (464, 640), (457, 640), (480, 640), (480, 640), (428, 640), (480, 640), (640, 480), (425, 640), (426, 640), (360, 640), (375, 500), (480, 640), (426, 640), (318, 500), (428, 640), (480, 640), (500, 375), (640, 480), (500, 347), (612, 612), (425, 640), (480, 640), (542, 640), (412, 640), (427, 640), (378, 500), (640, 425), (640, 512), (640, 399), (640, 480), (425, 640), (425, 640), (612, 612), (481, 640), (427, 640), (640, 427), (419, 640), (640, 428), (427, 640), (453, 640), (640, 425), (479, 640), (480, 640), (480, 640), (640, 457), (415, 640), (640, 512), (480, 640), (640, 480), (443, 640), (427, 640), (497, 640), (333, 500), (640, 427), (319, 640), (640, 425), (640, 427), (640, 480), (375, 500), (400, 640), (640, 468), (640, 425), (478, 640), (500, 332), (424, 640), (424, 640), (493, 640), (485, 640), (500, 333), (426, 640), (427, 640), (425, 640), (426, 640), (426, 640), (429, 640), (427, 640), (640, 480), (364, 640), (612, 612), (399, 640), (480, 640), (417, 640), (400, 500), (421, 640), (321, 640), (480, 640), (640, 427), (640, 427), (480, 640), (480, 640), (480, 640), (640, 428), (425, 640), (480, 640), (640, 480), (427, 640), (640, 414), (480, 640), (480, 640), (425, 640), (640, 558), (424, 640), (480, 640), (528, 640), (640, 509), (425, 640), (612, 612), (375, 500), (417, 500), (640, 427), (424, 640), (480, 640), (480, 640), (640, 480), (375, 500), (640, 430), (480, 640), (427, 640), (480, 640), (427, 640), (457, 640), (425, 640), (640, 427), (500, 375), (640, 481), (476, 640), (349, 640), (426, 640), (488, 640), (500, 333), (640, 480), (640, 444), (480, 640), (427, 640), (375, 500), (480, 640), (240, 320), (640, 457), (480, 640), (361, 640), (425, 640), (480, 640), (428, 640), (427, 640), (427, 640), (425, 640), (428, 640), (427, 640), (480, 640), (346, 640), (480, 640), (360, 640), (640, 559), (331, 500), (375, 500), (640, 359), (479, 640), (640, 457), (480, 640), (640, 480), (500, 375), (640, 427), (429, 640), (640, 480), (480, 640), (428, 640), (481, 640), (531, 640), (500, 375), (427, 640), (640, 480), (351, 640), (427, 640), (640, 480), (640, 459), (480, 640), (360, 640), (640, 480), (640, 640), (425, 640), (425, 640), (480, 640), (640, 480), (427, 640), (333, 500), (479, 640), (640, 480), (426, 640), (376, 640), (640, 426), (428, 640), (427, 640), (333, 500), (565, 640), (640, 480), (521, 640), (480, 640), (417, 640), (480, 640), (375, 500), (333, 500), (480, 640), (396, 500), (640, 317), (640, 425), (427, 640), (640, 480), (428, 640), (480, 640), (480, 640), (375, 500), (426, 640), (361, 640), (425, 640), (500, 375), (427, 640), (640, 427), (424, 640), (427, 640), (640, 427), (426, 640), (513, 640), (500, 375), (640, 437), (427, 640), (425, 640), (480, 640), (457, 640), (480, 640), (480, 640), (640, 600), (424, 640), (425, 640), (480, 640), (425, 640), (480, 640), (296, 500), (640, 478), (399, 640), (624, 640), (426, 640)]

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
    print(iter)
    coco.summarize_accuracy()


if __name__ == "__main__":
    # real_data = open_json_file("/home/jan/onspecta/cocoapi/annotations/instances_val2014.json")
    # for i in real_data["annotations"]:
    #    print(i)
    run_coco_test()
