# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import torch
import hashlib
import pkg_resources
from utils.profiling import *
from torch.autograd.profiler import profile
from pathlib import Path
from packaging import version
from utils.benchmark import *


class PyTorchRunner(Runner):
    """
    A class providing facilities to run PyTorch model (as pretrained torchvision model).
    """

    def __init__(self, model, disable_jit_freeze=False, example_inputs=None, func=None, skip_script=False):
        super().__init__()
        try:
            torch._C._aio_profiler_print()
            AIO = True
        except AttributeError:
            utils.advertise_aio("Torch")
            AIO = False

        torch.set_num_threads(get_intra_op_parallelism_threads())
        self._model = model
        self._func = func
        self._model.eval()
        self._frozen_script = None

        if disable_jit_freeze:
            if os.environ.get("TORCH_COMPILE") == "1":
                utils.print_goodbye_message_and_die(f"disable_jit_freeze={disable_jit_freeze} and TORCH_COMPILE=1 are mutually exclusive.")
            if AIO:
                utils.print_warning_message(
                    f"Running with disable_jit_freeze={disable_jit_freeze} - "
                    f"Ampere optimizations are not expected to work.")
        else:
            cached_dir = Path(os.path.dirname(os.path.realpath(__file__)) + "/cached")
            cached_path = cached_dir / f"{self._model._get_name()}_{hashlib.sha224(str(model).encode('utf-8')).hexdigest()}.pt"
            if os.environ.get("TORCH_COMPILE") == "1" and version.parse(pkg_resources.get_distribution("torch").version) >= version.parse("1.14"):
                # More natural comparison to version.parse("2.0") returns False for 2.0.0a0+git07156c4.dev, which is wrong.
                # There was never a PyTorch 1.14, so this comparison acts like comparing to 2.0, but works correctly for such edge cases.
                self.__frozen_script = torch.compile(self._model, backend="aio" if AIO else "inductor")
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
                        self.__frozen_script = torch.jit.freeze(torch.jit.script(self._model), preserved_attrs=[func])
                    else:
                        self.__frozen_script = torch.jit.freeze(torch.jit.script(self._model))
                except (torch.jit.frontend.UnsupportedNodeError, RuntimeError, SkipScript):
                    self.__frozen_script = torch.jit.freeze(torch.jit.trace(self._model, example_inputs))
                if not cached_dir.exists():
                    cached_dir.mkdir()
                torch.jit.save(self.__frozen_script, cached_path)
                print(f"Cached to file at {cached_path}")
        self._is_profiling = aio_profiler_enabled()

        print("\nRunning with PyTorch\n")

    def run(self, task_size: int, *args, **kwargs):
        """
        A function assigning values to input tensor, executing single pass over the network, measuring the time needed
        and finally returning the output.
        :return: dict, output dictionary with tensor names and corresponding output
        """

        def runner_func():
            start = time.time()
            output = model(*args, **kwargs)
            finish = time.time()

            self._start_times.append(start)
            self._finish_times.append(finish)
            self._workload_size.append(task_size)
            self._times_invoked += 1

            return output

        with torch.no_grad():
            if self.__frozen_script is None:
                model = self._model
            else:
                model = self.__frozen_script
            if self._func is not None:
                model = getattr(model, self._func)

            if self._is_profiling:
                with profile() as self.__profile:
                    output_tensor = runner_func()
            else:
                output_tensor = runner_func()

        return output_tensor

    def print_performance_metrics(self):
        if self._is_profiling:
            print(self.__profile.key_averages().table(sort_by='cpu_time_total', row_limit=50))
            torch._C._aio_profiler_print()
        return self.print_metrics()


class PyTorchRunnerV2(Runner):
    def __init__(self, model):
        super().__init__()
        try:
            torch._C._aio_profiler_print()
            utils.print_warning_message(
                f"Remember to compile your model with torch.jit / torch.compile for Ampere optimizations to work.")
        except AttributeError:
            utils.advertise_aio("Torch")

        torch.set_num_threads(get_intra_op_parallelism_threads())
        self._model = model

        self._do_profile = aio_profiler_enabled()

        self._times_invoked = 0
        self._start_times = list()
        self._finish_times = list()

        print("\nRunning with PyTorch\n")

    def run(self, task_size, *args, **kwargs):
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
            self._workload_size.append(task_size)
            self._times_invoked += 1

            return output

        with torch.no_grad():
            if self._do_profile:
                with profile() as self._profile:
                    return runner_func()
            else:
                return runner_func()

    def print_performance_metrics(self, batch_size, variable_input_lengths):
        if self._do_profile:
            print(self._profile.key_averages().table(sort_by='cpu_time_total', row_limit=50))
            torch._C._aio_profiler_print()
        return self.__print_performance_metrics()


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
