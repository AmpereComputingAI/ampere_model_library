# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import csv
import json
import torch
import utils.misc as utils
import time
import utils.benchmark as bench_utils
import hashlib
from utils.profiling import *
from torch.autograd.profiler import profile
from pathlib import Path


class PyTorchRunner:
    """
    A class providing facilities to run PyTorch model (as pretrained torchvision model).
    """

    def __init__(self, model, disable_jit_freeze=False, example_inputs=None, func=None, skip_script=False):
        try:
            torch._C._aio_profiler_print()
            AIO = True
        except AttributeError:
            utils.advertise_aio("Torch")
            AIO = False

        torch.set_num_threads(bench_utils.get_intra_op_parallelism_threads())
        self.__model = model.cuda()
        self.__func = func
        self.__model.eval()
        self.__frozen_script = None
        if disable_jit_freeze:
            if AIO:
                utils.print_warning_message(
                    f"Running with disable_jit_freeze={disable_jit_freeze} - Ampere optimizations are not expected to work.")
        else:
            cached_dir = Path(os.path.dirname(os.path.realpath(__file__)) + "/cached")
            cached_path = cached_dir / f"{self.__model._get_name()}_{hashlib.sha224(str(model).encode('utf-8')).hexdigest()}.pt"
            if cached_path.exists():
                self.__frozen_script = torch.jit.load(cached_path)
                print(f"Loaded from cached file at {cached_path}")
            else:
                try:
                    if skip_script:
                        raise SkipScript 
                    self.__frozen_script = torch.jit.freeze(torch.jit.script(self.__model))
                except (torch.jit.frontend.UnsupportedNodeError, SkipScript):
                    self.__frozen_script = torch.jit.freeze(torch.jit.trace(self.__model, example_inputs))
                if not cached_dir.exists():
                    cached_dir.mkdir()
                torch.jit.save(self.__frozen_script, cached_path)
                print(f"Cached to file at {cached_path}")

        self.__is_profiling = aio_profiler_enabled()

        if self.__frozen_script is not None:
            self.__model = self.__frozen_script

        if self.__func is not None:
            self.__model = getattr(self.__model, self.__func)
        self.__model = self.__model.cuda()

        self.__times_invoked = 0
        self.__start_times = list()
        self.__finish_times = list()
        self._mem_start_times = list()
        self._mem_finish_times = list()

        print("\nRunning with PyTorch\n")

    def run(self, input):
        """
        A function assigning values to input tensor, executing single pass over the network, measuring the time needed
        and finally returning the output.
        :return: dict, output dictionary with tensor names and corresponding output
        """

        def runner_func(model):
            if isinstance(input, tuple):
                fhfh
                start = time.time()
                output = model(*input)
                finish = time.time()
            elif isinstance(input, dict):
                start_mem = time.time()
                input_tensor = {name: val.cuda() for name, val in input.items()}
                torch.cuda.synchronize()
                finish_mem = time.time()
                start = time.time()
                output = model(**input_tensor)#, labels=input["input_ids"])
                torch.cuda.synchronize()
                finish = time.time()
            else:
                start_mem = time.time()
                input_tensor = input.cuda()
                torch.cuda.synchronize()
                finish_mem = time.time()
                start = time.time()
                output = model(input_tensor)
                torch.cuda.synchronize()
                finish = time.time()

            try:
                if type(output) is tuple:
                    a = time.time()
                    output = (out.cpu() for out in output)
                    b = time.time()
                elif type(output) is list:
                    a = time.time()
                    output = [{name: val.cpu() for name, val in out.items()} for out in output]
                    b = time.time()
                else:
                    a = time.time()
                    output = output.cpu()
                    b = time.time()
            except AttributeError:
                a = time.time()
                output = output.logits.cpu()
                b = time.time()

            self.__start_times.append(start)
            self.__finish_times.append(finish)
            self._mem_start_times.append(a-(finish_mem-start_mem))
            self._mem_finish_times.append(b)
            self.__times_invoked += 1

            return output

        with torch.no_grad():
            if self.__is_profiling:
                with profile() as self.__profile:
                    output_tensor = runner_func(self.__model)
            else:
                output_tensor = runner_func(self.__model)

        return output_tensor

    def print_performance_metrics(self, batch_size):
        perf = bench_utils.print_performance_metrics(
            self.__start_times, self.__finish_times, self.__times_invoked, batch_size
        )

        dump_dir = os.environ.get("RESULTS_DIR")
        if dump_dir is not None and len(self.__start_times) > 2:
            with open(f"{dump_dir}/meta_{os.getpid()}.json", "w") as f:
                json.dump({"batch_size": batch_size}, f)
            with open(f"{dump_dir}/{os.getpid()}_gpu.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(self.__start_times[2:])
                writer.writerow(self.__finish_times[2:])
            with open(f"{dump_dir}/{os.getpid()}_mem.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(self._mem_start_times[2:])
                writer.writerow(self._mem_finish_times[2:])

        if self.__is_profiling:
            print(self.__profile.key_averages().table(sort_by='cpu_time_total', row_limit=50))
            torch._C._aio_profiler_print()
        return perf

class SkipScript(Exception):
    pass
