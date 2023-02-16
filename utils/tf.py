# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import os
import csv
import time
import json
from datetime import datetime

import tensorflow as tf

import utils.benchmark as bench_utils
from utils.misc import advertise_aio


class TFProfiler:
    def __init__(self):
        self.__do_profile = os.getenv("PROFILER", "0") == "1"
        if self.__do_profile:
            options = tf.profiler.experimental.ProfilerOptions()
            time_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            tf.profiler.experimental.start(f"./profiler_output/tf2/{time_stamp}", options=options)

    def dump_maybe(self):
        if self.__do_profile:
            tf.profiler.experimental.stop()
            print("\nTo display TF profiler data run:\n  python3 -m tensorboard.main --logdir=./profiler_output/")


class TFFrozenModelRunner:
    """
    A class providing facilities to run TensorFlow frozen model (in frozen .pb format).
    """
    def __init__(self, path_to_model: str, output_names: list):
        """
        A function initializing runner by providing path to model and list of output names (can be easily checked with
        Netron app).
        :param path_to_model: str, eg. "ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
        :param output_names: list of str, eg. ["detection_classes:0", "detection_boxes:0"]
        """
        try:
            tf.AIO
        except AttributeError:
            advertise_aio("TensorFlow")

        self.__graph = self.__initialize_graph(path_to_model)
        self.__sess = tf.compat.v1.Session(
            config=self.__create_config(bench_utils.get_intra_op_parallelism_threads()),
            graph=self.__graph
        )
        self.__feed_dict = dict()
        self.__output_dict = {output_name: self.__graph.get_tensor_by_name(output_name) for output_name in output_names}

        self.__times_invoked = 0
        self.__start_times = list()
        self.__finish_times = list()

        self.__profiler = TFProfiler()

        print("\nRunning with TensorFlow\n")

    def __create_config(self, intra_threads: int, inter_threads=1):
        """
        A function creating TF config for given num of threads.
        :param intra_threads: int
        :param inter_threads: int
        :return: TensorFlow config
        """
        config = tf.compat.v1.ConfigProto()
        config.allow_soft_placement = True
        config.intra_op_parallelism_threads = intra_threads
        config.inter_op_parallelism_threads = inter_threads
        return config

    def __initialize_graph(self, path_to_model: str):
        """
        A function initializing TF graph from frozen .pb model.
        :param path_to_model: str
        :return: TensorFlow graph
        """
        graph = tf.compat.v1.Graph()
        with graph.as_default():
            graph_def = tf.compat.v1.GraphDef()
            with tf.compat.v1.gfile.GFile(path_to_model, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.compat.v1.import_graph_def(graph_def, name="")
        return graph

    def set_input_tensor(self, input_name: str, input_array):
        """
        A function assigning given numpy input array to the tensor under the provided input name.
        :param input_name: str, name of a input node in a model, eg. "image_tensor:0"
        :param input_array: numpy array with intended input
        """
        with tf.device('/gpu:0'):
            self.__feed_dict[self.__graph.get_tensor_by_name(input_name)] = tf.Variable(input_array)

    def run(self):
        """
        A function executing single pass over the network, measuring the time needed and returning the output.
        :return: dict, output dictionary with tensor names and corresponding output
        """

        start = time.time()
        output = self.__sess.run(self.__output_dict, self.__feed_dict)
        finish = time.time()

        self.__start_times.append(start)
        self.__finish_times.append(finish)
        self.__times_invoked += 1

        return output

    def print_performance_metrics(self, batch_size):
        """
        A function printing performance metrics on runs executed by the runner so far and then closing TF session.
        :param batch_size: int, batch size - if batch size was varying over the runs an average should be supplied
        """
        perf = bench_utils.print_performance_metrics(
            self.__start_times, self.__finish_times, self.__times_invoked, batch_size)
        if os.getenv("AIO_PROFILER", "0") == "1":
            tf.AIO.print_profile_data()
        self.__profiler.dump_maybe()
        self.__sess.close()

        dump_dir = os.environ.get("RESULTS_DIR")
        if dump_dir is not None and len(self.__start_times) > 2:
            with open(f"{dump_dir}/meta_{os.getpid()}.json", "w") as f:
                json.dump({"batch_size": batch_size}, f)
            with open(f"{dump_dir}/{os.getpid()}_gpu.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(self.__start_times[2:])
                writer.writerow(self.__finish_times[2:])

        return perf


class TFSavedModelRunner:
    """
    A class providing facilities to run TensorFlow saved model (in SavedModel format).
    """
    def __init__(self):
        """
        A function initializing runner.
        """
        try:
            tf.AIO
        except AttributeError:
            advertise_aio("TensorFlow")

        tf.config.threading.set_intra_op_parallelism_threads(bench_utils.get_intra_op_parallelism_threads())
        tf.config.threading.set_inter_op_parallelism_threads(1)

        self.model = None

        self.__times_invoked = 0
        self.__start_times = list()
        self.__finish_times = list()

        self.__profiler = TFProfiler()

        print("\nRunning with TensorFlow\n")

    def run(self, input):
        """
        A function assigning values to input tensor, executing single pass over the network, measuring the time needed
        and finally returning the output.
        :return: dict, output dictionary with tensor names and corresponding output
        """
        start = time.time()
        output = self.model(input)
        finish = time.time()

        self.__start_times.append(start)
        self.__finish_times.append(finish)
        self.__times_invoked += 1
        return output

    def print_performance_metrics(self, batch_size):
        """
        A function printing performance metrics on runs executed by the runner so far.
        :param batch_size: int, batch size - if batch size was varying over the runs an average should be supplied
        """
        perf = bench_utils.print_performance_metrics(
            self.__start_times, self.__finish_times, self.__times_invoked, batch_size)
        if os.getenv("AIO_PROFILER", "0") == "1":
            tf.AIO.print_profile_data()
        self.__profiler.dump_maybe()

        dump_dir = os.environ.get("RESULTS_DIR")
        if dump_dir is not None and len(self.__start_times) > 2:
            with open(f"{dump_dir}/meta_{os.getpid()}.json", "w") as f:
                json.dump({"batch_size": batch_size}, f)
            with open(f"{dump_dir}/{os.getpid()}.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(self.__start_times[2:])
                writer.writerow(self.__finish_times[2:])

        return perf
