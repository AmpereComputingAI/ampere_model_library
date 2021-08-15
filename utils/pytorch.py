import torchvision
import torch
import utils.misc as utils
import time
import utils.benchmark as bench_utils
import numpy as np
import sys


class PyTorchRunner:
    """
    A class providing facilities to run PyTorch model (as pretrained torchvision model).
    """

    def __init__(self, model: str):

        if model not in torchvision.models.__dict__:
            utils.print_goodbye_message_and_die(
                f"{model} not supported by torchvision!")

        self.__model = torchvision.models.__dict__[model](pretrained=True)
        self.__model.eval()
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

        with torch.no_grad():

            start = time.time()
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
