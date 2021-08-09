import torchvision
import torch
import utils.misc as utils
import time
import utils.benchmark as bench_utils
import numpy as np
import sys
np.set_printoptions(threshold=sys.maxsize)


class TFSavedModelRunner:
    """
    A class providing facilities to run TensorFlow saved model (in SavedModel format).
    """
    def __init__(self, path_to_model: str):
        """
        A function initializing runner by providing path to model directory.

        :param path_to_model: str, eg. "./path/to/yolo_saved_model/"
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


class PyTorchRunner:
    """
    A class providing facilities to run PyTorch model (as pretrained torchvision model).
    """

    def __init__(self, model: str):

        if model not in torchvision.models.__dict__:
            utils.print_goodbye_message_and_die(
                f"{model} not supported by torchvision!")

        self.__model = torchvision.models.__dict__[model](pretrained=True)
        self.__warm_up_run_latency = 0.0
        self.__total_inference_time = 0.0
        self.__times_invoked = 0


    def run(self, input):
        """
        A function assigning values to input tensor, executing single pass over the network, measuring the time needed
        and finally returning the output.

        :return: dict, output dictionary with tensor names and corresponding output
        """

        input_tensor = torch.from_numpy(input)
        output_tensor = self.__model(input_tensor)
        finish = time.time()
        output_tensor = output_tensor.detach().numpy()

        self.__total_inference_time += finish - start
        if self.__times_invoked == 0:
            self.__warm_up_run_latency += finish - start
        self.__times_invoked += 1

        return output_tensor

    def print_performance_metrics(self, batch_size):
        perf = bench_utils.print_performance_metrics(
            self.__warm_up_run_latency, self.__total_inference_time, self.__times_invoked, batch_size)
        return perf
