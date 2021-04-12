import utils.coco_utils as coco_utils
import utils.tf_utils as tf_utils
import tensorflow.compat.v1 as tf
import time
import os


def run_ssd_mn_v2_with_tf(number_of_runs=100, batch_size=1, shape=(640, 640)):
    coco = coco_utils.COCODataset(batch_size, images_filename_base="COCO_val2014_000000000000")

    runner = tf_utils.TFFrozenModelRunner(
        "ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb",
        ["detection_classes:0", "detection_boxes:0", "detection_scores:0", "num_detections:0"])

    # TODO: coco.get_possible_runs()
    for _ in range(int(number_of_runs / batch_size)):
        # start = time.time()
        runner.set_input_tensor("image_tensor:0", coco.get_input_array(shape))
        # print(time.time() - start)
        output = runner.run()
        print(output["detection_scores:0"])
        for i in range(batch_size):
            for d in range(int(output["num_detections:0"][i])):
                # if output["detection_scores:0"][i][d] < 0.6:
                #    continue
                coco.submit_bbox_prediction(
                    i,
                    coco.convert_bbox_to_coco_order(output["detection_boxes:0"][i][d] * shape[0], 1, 0, 3, 2),
                    int(output["detection_classes:0"][i][d])
                )

    coco.summarize_accuracy()
    runner.print_performance_metrics(batch_size)


if __name__ == "__main__":
    run_ssd_mn_v2_with_tf()
