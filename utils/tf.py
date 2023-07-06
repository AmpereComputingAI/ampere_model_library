# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC
from datetime import datetime
import tensorflow as tf
from utils.benchmark import *
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


class TFFrozenModelRunner(Runner):
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
        super().__init__()
        try:
            tf.AIO
        except AttributeError:
            advertise_aio("TensorFlow")

        self._graph = self._initialize_graph(path_to_model)
        self._sess = tf.compat.v1.Session(
            config=self._create_config(get_intra_op_parallelism_threads()),
            graph=self._graph
        )
        self._feed_dict = dict()
        self._output_dict = {output_name: self._graph.get_tensor_by_name(output_name) for output_name in output_names}

        self._profiler = TFProfiler()

        print("\nRunning with TensorFlow\n")

    def _create_config(self, intra_threads: int, inter_threads=1):
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

    def _initialize_graph(self, path_to_model: str):
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
        self._feed_dict[self._graph.get_tensor_by_name(input_name)] = input_array

    def run(self, task_size: int, *args, **kwargs):
        """
        A function executing single pass over the network, measuring the time needed and returning the output.
        :return: dict, output dictionary with tensor names and corresponding output
        """

        start = time.time()
        output = self._sess.run(self._output_dict, self._feed_dict)
        finish = time.time()

        self._start_times.append(start)
        self._finish_times.append(finish)
        self._workload_size.append(task_size)
        self._times_invoked += 1

        return output

    def print_performance_metrics(self):
        """
        A function printing performance metrics on runs executed by the runner so far and then closing TF session.
        """
        if os.getenv("AIO_PROFILER", "0") == "1":
            tf.AIO.print_profile_data()
        self._profiler.dump_maybe()
        self._sess.close()
        return self.__print_performance_metrics()


class TFSavedModelRunner(Runner):
    """
    A class providing facilities to run TensorFlow saved model (in SavedModel format).
    """
    def __init__(self):
        """
        A function initializing runner.
        """
        super().__init__()
        try:
            tf.AIO
        except AttributeError:
            advertise_aio("TensorFlow")

        tf.config.threading.set_intra_op_parallelism_threads(get_intra_op_parallelism_threads())
        tf.config.threading.set_inter_op_parallelism_threads(1)

        self.model = None
        self._profiler = TFProfiler()

        print("\nRunning with TensorFlow\n")

    def run(self, task_size: int, *args, **kwargs):
        """
        A function assigning values to input tensor, executing single pass over the network, measuring the time needed
        and finally returning the output.
        :return: dict, output dictionary with tensor names and corresponding output
        """
        start = time.time()
        output = self.model(*args, **kwargs)
        finish = time.time()

        self._start_times.append(start)
        self._finish_times.append(finish)
        self._workload_size.append(task_size)
        self._times_invoked += 1

        return output

    def print_performance_metrics(self):
        """
        A function printing performance metrics on runs executed by the runner so far.
        """
        if os.getenv("AIO_PROFILER", "0") == "1":
            tf.AIO.print_profile_data()
        self._profiler.dump_maybe()
        return self.__print_performance_metrics()
