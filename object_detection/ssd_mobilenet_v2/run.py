import utils.new_coco as coco_utils
import utils.tf as tf_utils
import tensorflow.compat.v1 as tf
import time
import os

TIMEOUT = 1200.0


def run_ssd_mn_v2_with_tf(number_of_runs=100, batch_size=1, shape=(640, 640)):
    coco = coco_utils.COCODataset(batch_size)

    runner = tf_utils.TFFrozenModelRunner(
        "ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb",
        ["detection_classes:0", "detection_boxes:0", "detection_scores:0", "num_detections:0"])

    # TODO: coco.get_possible_runs()
    start = time.time()
    while time.time() - start < TIMEOUT:
        # start = time.time()
        try:
            runner.set_input_tensor("image_tensor:0", coco.get_input_array(shape))
        except coco.OutOfCOCOImages:
            break
        # print(time.time() - start)
        output = runner.run()

        for i in range(batch_size):
            for d in range(int(output["num_detections:0"][i])):
                # if output["detection_scores:0"][i][d] < 0.6:
                #    continue
                #if int(output["detection_classes:0"][i][d]) == 1:
                #    print(output["detection_scores:0"][i][d])

                coco.submit_bbox_prediction(
                    i,
                    coco.convert_bbox_to_coco_order(output["detection_boxes:0"][i][d] * shape[0], 1, 0, 3, 2),
                    output["detection_scores:0"][i][d],
                    int(output["detection_classes:0"][i][d])
                )

    coco.summarize_accuracy()
    runner.print_performance_metrics(batch_size)


if __name__ == "__main__":
    run_ssd_mn_v2_with_tf()
