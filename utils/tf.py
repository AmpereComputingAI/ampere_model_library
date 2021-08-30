import os
import time
import tensorflow as tf
import utils.benchmark as bench_utils
from tensorflow.python.saved_model import tag_constants
from transformers import TFAutoModelForSequenceClassification
from utils.profiling import get_profile_path

from transformers import AutoConfig
from functools import wraps
tf.config.run_functions_eagerly(False)
import timeit
import random


def run_with_tf_optimizations(do_eager_mode: bool, use_xla: bool):
    def run_func(func):
        @wraps(func)
        def run_in_eager_mode(*args, **kwargs):
            return func(*args, **kwargs)

        @wraps(func)
        @tf.function()
        def run_in_graph_mode(*args, **kwargs):
            return func(*args, **kwargs)

        if do_eager_mode is True:
            assert (
                use_xla is False
            ), "Cannot run model in XLA, if `args.eager_mode` is set to `True`. Please set `args.eager_mode=False`."
            return run_in_eager_mode
        else:
            return run_in_graph_mode

    return run_func


class NLPModelRunner:
    """
    A class providing facilities to run TensorFlow frozen model (in frozen .pb format).
    """
    def __init__(self, model_name: str):

        tf.config.threading.set_intra_op_parallelism_threads(bench_utils.get_intra_op_parallelism_threads())
        tf.config.threading.set_inter_op_parallelism_threads(1)
        self.__model = TFAutoModelForSequenceClassification.from_pretrained(model_name)
        self.__warm_up_run_latency = 0.0
        self.__total_inference_time = 0.0
        self.__times_invoked = 0

        self.__model_name = model_name
        self.__config_dict = {
            model_name: AutoConfig.from_pretrained(model_name)
        }
        self.__model_class = "TF" + 'BertForSequenceClassification'
        self.__config = self.__config_dict[self.__model_name]
        self.__config.architectures[0] = self.__model_class
        self.__transformers_module = __import__("transformers", fromlist=[self.__model_class])
        self.__model_cls = getattr(self.__transformers_module, self.__model_class)
        self.__model1 = self.__model_cls(self.__config)

    def run(self, input):
        # ====================APPROACH 2=========================
        # model_class = "TF" + 'BertForSequenceClassification'
        # config = self.__config_dict[self.__model_name]
        # transformers_module = __import__("transformers", fromlist=[model_class])
        # model_cls = getattr(transformers_module, model_class)
        # print(config)
        # model1 = model_cls(config)
        # print(model1)
        # print(type(model1))
        #
        # print('PRINTING MODEL')
        # print(type(model))
        # print(model)
        #
        #
        # @run_with_tf_optimizations(False, False)
        # def test():
        #     print('here 3')
        #     return model(input, training=False)
        #
        #
        #
        # model = test
        # print('here 2')
        # start = time.time()
        # paraphrase_classification_logits = model()
        # print('here 4')
        # finish = time.time()
        # print('here 4')
        # ====================APPROACH 2=========================


        # writer = tf.summary.create_file_writer("/onspecta/dev/mz/natural_language_processing/test")

        1
        25
        28996

        def random_input_ids(batch_size: int, sequence_length: int, vocab_size: int) -> ["tf.Tensor"]:
            rng = random.Random()
            values = [rng.randint(0, vocab_size - 1) for i in range(batch_size * sequence_length)]
            return tf.constant(values, shape=(batch_size, sequence_length), dtype=tf.int32)

        inputs_ids = random_input_ids(1, 25, 28996)

        print('COMPARING MODELS')

        print(self.__model1)
        print(type(self.__model1))

        print(self.__model)
        print(type(self.__model))

        print(self.__model_class)
        # print(self.__config)
        print(self.__transformers_module)
        print(self.__model_cls)

        print(self.__config)

        @tf.function(experimental_compile=False)
        def test():
            paraphrase_classification_logits = self.__model1(input, training=False)[0]
            # with writer.as_default():
            #     tf.summary.scalar()
            return paraphrase_classification_logits

        def test1():
            paraphrase_classification_logits = self.__model(input)[0]
            # with writer.as_default():
            #     tf.summary.scalar()
            return paraphrase_classification_logits


        def test2():
            paraphrase_classification_logits = self.__model1(inputs_ids, training=False)[0]
            # with writer.as_default():
            #     tf.summary.scalar()
            return paraphrase_classification_logits

        # options = tf.profiler.experimental.ProfilerOptions(
        #     host_tracer_level=3,
        #     python_tracer_level=1,
        #     device_tracer_level=1
        # )
        # tf.profiler.experimental.start('/onspecta/dev/mz/natural_language_processing/test', options=options)

        # _ = model1(input)[0]
        # _ = self.__model(input)[0]
        _ = test2()

        # start = time.time()
        # paraphrase_classification_logits = self.__model(input)[0]
        # finish = time.time()

        runtimes = timeit.repeat(
            test,
            repeat=10,
            number=10,
        )

        yoooo = min(runtimes) / 10.0

        start = time.time()
        output = tf.nn.softmax(test(), axis=1).numpy()
        finish = time.time()
        # tf.profiler.experimental.stop()

        self.__total_inference_time += yoooo
        if self.__times_invoked == 0:
            self.__warm_up_run_latency += finish - start
        self.__times_invoked += 1

        return output

    def print_performance_metrics(self, batch_size):

        perf = bench_utils.print_performance_metrics(
            self.__warm_up_run_latency, self.__total_inference_time, self.__times_invoked, batch_size)
        return perf


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
        self.__graph = self.__initialize_graph(path_to_model)
        self.__sess = tf.compat.v1.Session(
            config=self.__create_config(bench_utils.get_intra_op_parallelism_threads()),
            graph=self.__graph
        )
        self.__feed_dict = dict()
        self.__output_dict = {output_name: self.__graph.get_tensor_by_name(output_name) for output_name in output_names}
        self.__warm_up_run_latency = 0.0
        self.__total_inference_time = 0.0
        self.__times_invoked = 0

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
        self.__feed_dict[self.__graph.get_tensor_by_name(input_name)] = input_array

    def run(self):
        """
        A function executing single pass over the network, measuring the time needed and returning the output.

        :return: dict, output dictionary with tensor names and corresponding output
        """
        start = time.time()
        output = self.__sess.run(self.__output_dict, self.__feed_dict)
        finish = time.time()
        self.__total_inference_time += finish - start
        if self.__times_invoked == 0:
            self.__warm_up_run_latency += finish - start
        self.__times_invoked += 1
        return output

    def print_performance_metrics(self, batch_size):
        """
        A function printing performance metrics on runs executed by the runner so far and then closing TF session.

        :param batch_size: int, batch size - if batch size was varying over the runs an average should be supplied
        """
        perf = bench_utils.print_performance_metrics(
            self.__warm_up_run_latency, self.__total_inference_time, self.__times_invoked, batch_size)
        self.__sess.close()
        return perf


class TFSavedModelRunner:
    """
    A class providing facilities to run TensorFlow saved model (in SavedModel format).
    """
    def __init__(self, path_to_model: str):
        """
        A function initializing runner by providing path to model directory.

        :param path_to_model: str, eg. "./ugabuga/yolo_saved_model/"
        """
        tf.config.threading.set_intra_op_parallelism_threads(bench_utils.get_intra_op_parallelism_threads())
        tf.config.threading.set_inter_op_parallelism_threads(1)
        self.__saved_model_loaded = tf.saved_model.load(path_to_model, tags=[tag_constants.SERVING])
        self.__model = self.__saved_model_loaded.signatures['serving_default']
        self.__warm_up_run_latency = 0.0
        self.__total_inference_time = 0.0
        self.__times_invoked = 0

    def run(self, input):
        """
        A function assigning values to input tensor, executing single pass over the network, measuring the time needed
        and finally returning the output.

        :return: dict, output dictionary with tensor names and corresponding output
        """
        start = time.time()
        output = self.__model(input)
        finish = time.time()
        self.__total_inference_time += finish - start
        if self.__times_invoked == 0:
            self.__warm_up_run_latency += finish - start
        self.__times_invoked += 1
        return output

    def print_performance_metrics(self, batch_size):
        """
        A function printing performance metrics on runs executed by the runner so far.

        :param batch_size: int, batch size - if batch size was varying over the runs an average should be supplied
        """
        perf = bench_utils.print_performance_metrics(
            self.__warm_up_run_latency, self.__total_inference_time, self.__times_invoked, batch_size)
        return perf
