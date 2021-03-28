import utils.coco_utils as coco_utils
import tensorflow.compat.v1 as tf
import time
import os


INTRA_OP_PARALLELISM_THREADS = None


def create_config(intra_threads: int, inter_threads=1):
    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.intra_op_parallelism_threads = intra_threads
    config.inter_op_parallelism_threads = inter_threads
    return config


def initialize_graph(path_to_model: str):
    graph = tf.Graph()
    with graph.as_default():
        graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_model, 'rb') as fid:
            serialized_graph = fid.read()
            graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(graph_def, name="")
    return graph


def get_intra_op_parallelism_threads():
    global INTRA_OP_PARALLELISM_THREADS
    if not INTRA_OP_PARALLELISM_THREADS:
        try:
            INTRA_OP_PARALLELISM_THREADS = int(os.environ["OMP_NUM_THREADS"])
        except KeyError:
            INTRA_OP_PARALLELISM_THREADS = int(os.environ["DLS_NUM_THREADS"])
        finally:
            if not INTRA_OP_PARALLELISM_THREADS:
                coco_utils.print_goodbye_message_and_die("Number of threads to run with is not set!")
    return INTRA_OP_PARALLELISM_THREADS


def print_benchmark_metrics(first_run_latency, total_inference_time, number_of_runs, batch_size):
    latency_in_seconds = (total_inference_time - first_run_latency) / (number_of_runs - 1)
    latency_in_ms = latency_in_seconds * 1000
    instances_per_second = batch_size / latency_in_seconds
    print("Latency: {:.2f} ms".format(latency_in_ms))
    print("Throughput: {:.2f} ips".format(instances_per_second))


class TensorFlowRunner:

    def __init__(self, path_to_model: str, output_names: list):
        self.graph = initialize_graph(path_to_model)
        self.sess = tf.Session(config=create_config(get_intra_op_parallelism_threads()), graph=self.graph)
        self.feed_dict = {}
        self.output_dict = {output_name: self.graph.get_tensor_by_name(output_name) for output_name in output_names}
        self.first_run_latency = 0.0
        self.total_inference_time = 0.0
        self.times_invoked = 0

    def set_input_tensor(self, input_name: str, input_array):
        self.feed_dict[self.graph.get_tensor_by_name(input_name)] = input_array

    def run(self):
        start = time.time()
        print(self.feed_dict)
        output = self.sess.run(self.output_dict, self.feed_dict)
        finish = time.time()
        self.total_inference_time += finish - start
        if self.times_invoked == 0:
            self.first_run_latency += finish - start
        self.times_invoked += 1
        return output


def run_ssd_mn_v2_with_tf(batch_size=1):
    coco_dataset = coco_utils.COCODataset(batch_size, (640, 640))
    runner = TensorFlowRunner("ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb",
                              ["detection_classes:0", "detection_boxes:0", "detection_scores:0", "num_detections:0"])

    for _ in range(1000):
        runner.set_input_tensor("image_tensor:0", coco_dataset.get_input_array())
        _ = runner.run()
    print_benchmark_metrics(runner.first_run_latency, runner.total_inference_time, runner.times_invoked, batch_size)
    sess.close()


if __name__ == "__main__":
    run_ssd_mn_v2_with_tf()
