# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import os
import csv
import time
import tensorflow as tf
import utils.benchmark as bench_utils
from utils.misc import advertise_aio


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
        try:
            tf.AIO
        except AttributeError:
            advertise_aio("TensorFlow")

        self.__interpreter = tf.compat.v1.lite.Interpreter(
            model_path=path_to_model, num_threads=bench_utils.get_intra_op_parallelism_threads())
        self.__interpreter.allocate_tensors()
        self.input_details = self.__interpreter.get_input_details()
        self.output_details = self.__interpreter.get_output_details()
        self.__warm_up_run_latency = 0.0
        self.__total_inference_time = 0.0
        self.__times_invoked = 0

        self.__start_times = list()
        self.__finish_times = list()

        print("\nRunning with TensorFlow Lite\n")

    def set_input_tensor(self, input_index: int, input_array):
        """
        A function assigning given numpy input array to the tensor under the provided input index.

        :param input_index: int, index of the input node in a model (can be obtained by accessing self.input_details)
        :param input_array: numpy array with intended input
        """
        self.__interpreter.set_tensor(input_index, input_array)

    def get_output_tensor(self, output_index):
        """
        A function returning an array with model's output available under the provided output index.

        :param output_index: int, index of the output node in a model (can be obtained by accessing self.output_details)
        :return: output tensor available under the supplied index
        """
        return self.__interpreter.get_tensor(output_index)

    def run(self):
        """
        A function executing single pass over the network, measuring the time needed and number of passes.

        :return: dict, output dictionary with tensor names and corresponding output
        """
        start = time.time()
        self.__interpreter.invoke()
        finish = time.time()
        self.__total_inference_time += finish - start
        if self.__times_invoked == 0:
            self.__warm_up_run_latency += finish - start
        else:
            self.__start_times.append(start)
            self.__finish_times.append(finish)
        self.__times_invoked += 1

    def print_performance_metrics(self, batch_size):
        """
        A function printing performance metrics on runs executed by the runner so far.

        :param batch_size: int, batch size - if batch size was varying over the runs an average should be supplied
        """
        perf = bench_utils.print_performance_metrics(
            self.__warm_up_run_latency, self.__total_inference_time, self.__times_invoked, batch_size)
        if os.getenv("AIO_PROFILER", "0") == "1":
            tf.AIO.print_profile_data()

        dump_dir = os.environ.get("RESULTS_DIR")
        if dump_dir is not None:
            with open(f"{dump_dir}/{os.getpid()}.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(self.__start_times)
                writer.writerow(self.__finish_times)

        return perf
