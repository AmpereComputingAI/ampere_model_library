# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import torch
import utils.misc as utils
import time
import utils.benchmark as bench_utils
import hashlib
import pkg_resources
from utils.profiling import *
from torch.autograd.profiler import profile
from pathlib import Path
from packaging import version


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
        self.__model = model
        self.__func = func
        self.__model.eval()
        self.__frozen_script = None

        if disable_jit_freeze:
            if os.environ.get("TORCH_COMPILE") == "1":
                utils.print_goodbye_message_and_die(f"disable_jit_freeze={disable_jit_freeze} and TORCH_COMPILE=1 are mutually exclusive.")
            if AIO:
                utils.print_warning_message(
                    f"Running with disable_jit_freeze={disable_jit_freeze} - "
                    f"Ampere optimizations are not expected to work.")
        else:
            cached_dir = Path(os.path.dirname(os.path.realpath(__file__)) + "/cached")
            cached_path = cached_dir / f"{self.__model._get_name()}_{hashlib.sha224(str(model).encode('utf-8')).hexdigest()}.pt"
            if os.environ.get("TORCH_COMPILE") == "1" and version.parse(pkg_resources.get_distribution("torch").version) >= version.parse("1.14"):
                # More natural comparison to version.parse("2.0") returns False for 2.0.0a0+git07156c4.dev, which is wrong.
                # There was never a PyTorch 1.14, so this comparison acts like comparing to 2.0, but works correctly for such edge cases.
                self.__frozen_script = torch.compile(self.__model, backend="aio" if AIO else "inductor")
            elif os.environ.get("TORCH_COMPILE") == "1" and not version.parse(pkg_resources.get_distribution("torch").version) >= version.parse("1.14"):
                utils.print_goodbye_message_and_die(f"TORCH_COMPILE=1 set, but installed PyTorch version is {pkg_resources.get_distribution('torch').version}. PyTorch version must be at least 2.0.0 to use torch.compile().")
            elif cached_path.exists():
                self.__frozen_script = torch.jit.load(cached_path)
                print(f"Loaded from cached file at {cached_path}")
            else:
                try:
                    if skip_script:
                        raise SkipScript 
                    if func:
                        self.__frozen_script = torch.jit.freeze(torch.jit.script(self.__model), preserved_attrs=[func])
                    else:
                        self.__frozen_script = torch.jit.freeze(torch.jit.script(self.__model))
                except (torch.jit.frontend.UnsupportedNodeError, RuntimeError, SkipScript):
                    self.__frozen_script = torch.jit.freeze(torch.jit.trace(self.__model, example_inputs))
                if not cached_dir.exists():
                    cached_dir.mkdir()
                torch.jit.save(self.__frozen_script, cached_path)
                print(f"Cached to file at {cached_path}")
        self.__is_profiling = aio_profiler_enabled()

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
            if isinstance(input, tuple):
                start = time.time()
                output = model(*input)
                finish = time.time()
            elif isinstance(input, dict):
                start = time.time()
                output = model(**input)
                finish = time.time()
            else:
                start = time.time()
                output = model(input)
                finish = time.time()

            self.__start_times.append(start)
            self.__finish_times.append(finish)
            self.__times_invoked += 1

            return output

        with torch.no_grad():
            if self.__frozen_script is None:
                model = self.__model
            else:
                model = self.__frozen_script
            if self.__func is not None:
                model = getattr(model, self.__func)

            if self.__is_profiling:
                with profile() as self.__profile:
                    output_tensor = runner_func(model)
            else:
                output_tensor = runner_func(model)

        return output_tensor

    def print_performance_metrics(self, batch_size):
        perf = bench_utils.print_performance_metrics(
            self.__start_times, self.__finish_times, self.__times_invoked, batch_size
        )

        bench_utils.dump_csv_results(batch_size, self.__start_times, self.__finish_times)

        if self.__is_profiling:
            print(self.__profile.key_averages().table(sort_by='cpu_time_total', row_limit=50))
            torch._C._aio_profiler_print()
        return perf


class PyTorchRunnerV2:
    def __init__(self, model):
        try:
            torch._C._aio_profiler_print()
            utils.print_warning_message(
                f"Remember to compile your model with torch.jit / torch.compile for Ampere optimizations to work.")
        except AttributeError:
            utils.advertise_aio("Torch")

        torch.set_num_threads(bench_utils.get_intra_op_parallelism_threads())
        self._model = model

        self._do_profile = aio_profiler_enabled()

        self._times_invoked = 0
        self._start_times = list()
        self._finish_times = list()

        print("\nRunning with PyTorch\n")

    def run(self, *args, **kwargs):
        """
        A function assigning values to input tensor, executing single pass over the network, measuring the time needed
        and finally returning the output.
        :return: dict, output dictionary with tensor names and corresponding output
        """

        def runner_func():
            start = time.time()
            output = self._model(*args, **kwargs)
            finish = time.time()

            self._start_times.append(start)
            self._finish_times.append(finish)
            self._times_invoked += 1

            return output

        with torch.no_grad():
            if self._do_profile:
                with profile() as self._profile:
                    return runner_func()
            else:
                return runner_func()

    def print_performance_metrics(self, batch_size):
        if self._do_profile:
            print(self._profile.key_averages().table(sort_by='cpu_time_total', row_limit=50))
            torch._C._aio_profiler_print()
        return bench_utils.print_performance_metrics(
            self._start_times, self._finish_times, self._times_invoked, batch_size
        )


def check_if_cached(model):
    cached_dir = Path(os.path.dirname(os.path.realpath(__file__)), "torch_jit_cache")
    if not cached_dir.exists():
        cached_dir.mkdir()
    cached_path = cached_dir / f"{hashlib.sha224(str(model).encode('utf-8')).hexdigest()}.pt"
    return cached_path.exists(), cached_path


def load_from_cache_or_apply(model, conversion):
    is_cached, cached_path = check_if_cached(model)
    if is_cached:
        print(f"Loading from cache ...")
        return torch.jit.load(cached_path)
    else:
        model = torch.jit.freeze(conversion())
        torch.jit.save(model, cached_path)
        print(f"Cached at {cached_path}")
        return model


def apply_jit_script(model):
    return load_from_cache_or_apply(model, lambda: torch.jit.script(model))


def apply_jit_trace(model, example_inputs):
    return load_from_cache_or_apply(model, lambda: torch.jit.trace(model, example_inputs))


class SkipScript(Exception):
    pass
