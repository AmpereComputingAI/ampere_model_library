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

    def __init__(self, model, jit_freeze=False):
        torch.set_num_threads(bench_utils.get_intra_op_parallelism_threads())
        self.__model = model
        self.__model.eval()
        self.__jit_freeze = jit_freeze
        if self.__jit_freeze:
            self.__model_script = torch.jit.script(self.__model)
            self.__frozen_script = torch.jit.freeze(self.__model_script)

        self.__warm_up_run_latency = 0.0
        self.__total_inference_time = 0.0
        self.__times_invoked = 0

        print("\nRunning with PyTorch\n")

    def run(self, input):
        """
        A function assigning values to input tensor, executing single pass over the network, measuring the time needed
        and finally returning the output.
        :return: dict, output dictionary with tensor names and corresponding output
        """

        with torch.no_grad():
            if self.__jit_freeze:
                start = time.time()
                output_tensor = self.__frozen_script(input)
                finish = time.time()
            else:
                if type(input) == tuple:
                    start = time.time()
                    output_tensor = self.__model(*input)
                    finish = time.time()
                else:
                    start = time.time()
                    output_tensor = self.__model(input)
                    finish = time.time()

        self.__total_inference_time += finish - start
        if self.__times_invoked == 0:
            self.__warm_up_run_latency += finish - start
        self.__times_invoked += 1

        return output_tensor

    def print_performance_metrics(self, batch_size):
        perf = bench_utils.print_performance_metrics(
            self.__warm_up_run_latency, self.__total_inference_time, self.__times_invoked, batch_size)
        if os.getenv("AIO_PROFILER", "0") == "1":
            torch.AIO.print_profile_data()
        return perf
