import os
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

    def __init__(self, model, disable_jit_freeze=False):
        torch.set_num_threads(bench_utils.get_intra_op_parallelism_threads())
        self.__model = model
        self.__model.eval()
        self.__frozen_script = None
        if not disable_jit_freeze:
            self.__frozen_script = torch.jit.freeze(torch.jit.script(self.__model))

        self.__warm_up_run_latency = 0.0
        self.__total_inference_time = 0.0
        self.__times_invoked = 0

        self.__start_times = list()
        self.__finish_times = list()

        print("\nRunning with PyTorch\n")

    def run(self, input):
        """
        A function assigning values to input tensor, executing single pass over the network, measuring the time needed
        and finally returning the output.
        :return: dict, output dictionary with tensor names and corresponding output
        """

        def runner_func(model):
            if type(input) == tuple:
                start = time.time()
                output = model(*input)
                finish = time.time()
            else:
                start = time.time()
                output = model(input)
                finish = time.time()

            self.__total_inference_time += finish - start
            if self.__times_invoked == 0:
                self.__warm_up_run_latency += finish - start
            else:
                self.__start_times.append(start)
                self.__finish_times.append(finish)
            self.__times_invoked += 1

            return output

        with torch.no_grad():
            if self.__frozen_script is not None:
                output_tensor = runner_func(self.__frozen_script)
            else:
                output_tensor = runner_func(self.__model)

        return output_tensor

    def print_performance_metrics(self, batch_size):
        perf = bench_utils.print_performance_metrics(
            self.__warm_up_run_latency, self.__total_inference_time, self.__times_invoked, batch_size)
        if os.getenv("AIO_PROFILER", "0") == "1":
            torch.AIO.print_profile_data()

        dump_dir = os.environ.get("RESULTS_DIR")
        if dump_dir is not None:
            with open(f"{dump_dir}/{os.getpid()}.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(self.__start_times)
                writer.writerow(self.__finish_times)

        return perf
