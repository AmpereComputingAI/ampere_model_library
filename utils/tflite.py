import os
import time
import tensorflow.compat.v1 as tf
import utils.benchmark as bench_utils


class TFLiteRunner:
    """
    A class providing facilities to run TensorFlow Lite model (in .tflite format).
    """
    def __init__(self, path_to_model: str):
        """
        A function initializing runner.

        :param path_to_model: str, eg. "ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
        :param output_names: list of str, eg. ["detection_classes:0", "detection_boxes:0"]
        """
        self.__interpreter = tf.lite.Interpreter(
            model_path=path_to_model, num_threads=bench_utils.get_intra_op_parallelism_threads())
        self.__interpreter.allocate_tensors()
        self.input_details = self.__interpreter.get_input_details()
        self.output_details = self.__interpreter.get_output_details()
        self.__warm_up_run_latency = 0.0
        self.__total_inference_time = 0.0
        self.__times_invoked = 0

    def set_input_tensor(self, input_index: int, input_array):
        """
        A function assigning given numpy input array to the tensor under the provided input name.

        :param input_index: str, name of a input node in a model, eg. "image_tensor:0"
        :param input_array: numpy array with intended input
        """
        self.__interpreter.set_tensor(input_index, input_array)

    def get_output_tensor(self, output_index):
        return self.__interpreter.get_tensor(output_index)

    def run(self):
        """
        A function executing single pass over the network, measuring the time need and returning the output.

        :return: dict, output dictionary with tensor names and corresponding output
        """
        start = time.time()
        self.__interpreter.invoke()
        finish = time.time()
        self.__total_inference_time += finish - start
        if self.__times_invoked == 0:
            self.__warm_up_run_latency += finish - start
        self.__times_invoked += 1

    def print_performance_metrics(self, batch_size):
        """
        A function printing performance metrics on runs executed by the runner so far and then closing TF session.

        :param batch_size: int, batch size - if batch size was varying over the runs an average should be supplied
        """
        bench_utils.print_performance_metrics(
            self.__warm_up_run_latency, self.__total_inference_time, self.__times_invoked, batch_size)
