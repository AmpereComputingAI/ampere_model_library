# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import tensorflow as tf
from utils.benchmark import *
from utils.misc import advertise_aio


class TFLiteRunner(Runner):
    """
    A class providing facilities to run TensorFlow Lite model (in .tflite format).
    """

    def __init__(self, path_to_model: str):
        """
        A function initializing runner.

        :param path_to_model: str, eg. "ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
        """
        super().__init__()
        try:
            tf.AIO
        except AttributeError:
            advertise_aio("TensorFlow")

        self._interpreter = tf.compat.v1.lite.Interpreter(
            model_path=path_to_model, num_threads=get_intra_op_parallelism_threads())
        self._interpreter.allocate_tensors()
        self.input_details = self._interpreter.get_input_details()
        self.output_details = self._interpreter.get_output_details()

        print("\nRunning with TensorFlow Lite\n")

    def set_input_tensor(self, input_index: int, input_array):
        """
        A function assigning given numpy input array to the tensor under the provided input index.

        :param input_index: int, index of the input node in a model (can be obtained by accessing self.input_details)
        :param input_array: numpy array with intended input
        """
        self._interpreter.set_tensor(input_index, input_array)

    def get_output_tensor(self, output_index):
        """
        A function returning an array with model's output available under the provided output index.

        :param output_index: int, index of the output node in a model (can be obtained by accessing self.output_details)
        :return: output tensor available under the supplied index
        """
        return self._interpreter.get_tensor(output_index)

    def run(self, task_size: int, *args, **kwargs):
        """
        A function executing single pass over the network, measuring the time needed and number of passes.

        :return: dict, output dictionary with tensor names and corresponding output
        """
        start = time.time()
        self._interpreter.invoke()
        finish = time.time()

        self._times_invoked += 1
        self._start_times.append(start)
        self._finish_times.append(finish)
        self._workload_size.append(task_size)

    def print_performance_metrics(self):
        """
        A function printing performance metrics on runs executed by the runner so far.
        """
        if os.getenv("AIO_PROFILER", "0") == "1":
            tf.AIO.print_profile_data()

        return self.print_metrics()
