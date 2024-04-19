# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import os
import time
import torch
import types
import hashlib
import pkg_resources
from utils.profiling import aio_profiler_enabled
from torch.autograd.profiler import profile
from pathlib import Path
from packaging import version
from contextlib import nullcontext
from utils.benchmark import Runner, get_intra_op_parallelism_threads
import utils.misc as utils


class PyTorchRunner(Runner):
    """
    A class providing facilities to run PyTorch model (as pretrained torchvision model).
    """

    def __init__(self,
                 model,
                 disable_jit_freeze=False, example_inputs=None, func=None, skip_script=False, throughput_only=False):
        super().__init__(throughput_only)
        AIO = '_aio_profiler_print' in dir(torch._C)
        if AIO:
            utils.print_warning_message(
                "Remember to compile your model with torch.jit / torch.compile for Ampere optimizations to work.")
            utils.check_memory_settings()
        else:
            utils.advertise_aio("Torch")

        torch.set_num_threads(get_intra_op_parallelism_threads())
        self._model = model
        self._func = func
        self._model.eval()
        self._frozen_script = None

        self._do_autocast = os.environ.get("ENABLE_BF16_X86") == "1"
        self._gpu_autocast = os.environ.get("ENABLE_AUTOCAST_GPU") == "1"
        self._gpu = torch.cuda.is_available()

        if os.environ.get("IPEX_OPTIMIZE") == "1":
            import intel_extension_for_pytorch as ipex
            dtype = torch.bfloat16 if self._do_autocast else torch.float32
            self._model = ipex.optimize(self._model, dtype=dtype)
            disable_jit_freeze = True

        if disable_jit_freeze:
            if os.environ.get("TORCH_COMPILE") == "1":
                utils.print_goodbye_message_and_die(
                    f"disable_jit_freeze={disable_jit_freeze} and TORCH_COMPILE=1 are mutually exclusive.")
            if AIO:
                utils.print_warning_message(f"Running with disable_jit_freeze={disable_jit_freeze} - Ampere "
                                            f"optimizations are not expected to work.")
        else:
            cached_dir = Path(os.path.dirname(os.path.realpath(__file__)) + "/cached")
            cached_path = (cached_dir /
                           f"{self._model._get_name()}_{hashlib.sha224(str(model).encode('utf-8')).hexdigest()}.pt")
            if os.environ.get("TORCH_COMPILE") == "1" and version.parse(
                    pkg_resources.get_distribution("torch").version) >= version.parse("1.14"):
                # More natural comparison to version.parse("2.0") returns False for 2.0.0a0+git07156c4.dev, which is
                # wrong. There was never a PyTorch 1.14, so this comparison acts like comparing to 2.0, but works
                # correctly for such edge cases.
                self._frozen_script = torch.compile(self._model, backend="aio" if AIO else "inductor",
                                                    options={"modelname": self._model._get_name()} if AIO else {})
            elif os.environ.get("TORCH_COMPILE") == "1" and not version.parse(
                    pkg_resources.get_distribution("torch").version) >= version.parse("1.14"):
                utils.print_goodbye_message_and_die(
                    f"TORCH_COMPILE=1 set, but installed PyTorch version is "
                    f"{pkg_resources.get_distribution('torch').version}. PyTorch version must be at least 2.0.0 "
                    f"to use torch.compile().")
            elif cached_path.exists():
                self._frozen_script = torch.jit.load(cached_path)
                print(f"Loaded from cached file at {cached_path}")
            else:
                try:
                    if skip_script:
                        raise SkipScript
                    if func:
                        self._frozen_script = torch.jit.freeze(torch.jit.script(self._model), preserved_attrs=[func])
                    else:
                        self._frozen_script = torch.jit.freeze(torch.jit.script(self._model))
                except (torch.jit.frontend.UnsupportedNodeError, RuntimeError, SkipScript):
                    self._frozen_script = torch.jit.freeze(torch.jit.trace(self._model, example_inputs))
                if not cached_dir.exists():
                    cached_dir.mkdir()
                torch.jit.save(self._frozen_script, cached_path)
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
            if self._do_autocast:
                context = torch.cpu.amp.autocast()
            elif self._gpu_autocast:
                context = torch.cuda.amp.autocast()
            else:
                context = nullcontext()

            with context:
                start = time.time()
                output = model(*args, **kwargs)
                finish = time.time()
                if self._gpu:
                    torch.cuda.synchronize()
                    finish = time.time()

            self._start_times.append(start)
            self._finish_times.append(finish)
            self._workload_size.append(task_size)
            self._times_invoked += 1
            return output

        with torch.no_grad():
            if self._frozen_script is None:
                model = self._model
            else:
                model = self._frozen_script
            if self._func is not None:
                model = getattr(model, self._func)

            if self._is_profiling:
                with profile() as self._profile:
                    output_tensor = runner_func()
            else:
                output_tensor = runner_func()

        return output_tensor

    def print_performance_metrics(self):
        if self._is_profiling:
            print(self._profile.key_averages().table(sort_by='cpu_time_total', row_limit=50))
            torch._C._aio_profiler_print()
        return self.print_metrics()


class PyTorchRunnerV2(Runner):
    def __init__(self, model, throughput_only=False):
        super().__init__(throughput_only)
        AIO = '_aio_profiler_print' in dir(torch._C)
        if AIO:
            utils.print_warning_message(
                "Remember to compile your model with torch.jit / torch.compile for Ampere optimizations to work.")
            utils.check_memory_settings()
        else:
            utils.advertise_aio("Torch")

        torch.set_num_threads(get_intra_op_parallelism_threads())
        self._do_autocast = os.environ.get("ENABLE_BF16_X86") == "1"
        self._gpu_autocast = os.environ.get("ENABLE_AUTOCAST_GPU") == "1"
        self._gpu = torch.cuda.is_available()
        if os.environ.get("IPEX_OPTIMIZE") == "1":
            # when using PyTorchRunnerV2, IPEX optimization should be handled in model's run file - it is here just to
            # ensure it has been accounted for in the run file. I.e. when applied a second time on model object,
            # ipex.optimize() will just pass, but for example if it is applied here per user's request on a torch
            # scripted model coming as argument to this function, ipex.optimize() will likely fail - and this issue
            # shouldn't be solved here but in the model's run file
            import intel_extension_for_pytorch as ipex
            dtype = torch.bfloat16 if self._do_autocast else torch.float32
            model = ipex.optimize(model, dtype=dtype)
        self._model = model

        self._do_profile = aio_profiler_enabled()

        print("\nRunning with PyTorch\n")

    def run(self, task_size=None, *args, **kwargs):
        """
        A function assigning values to input tensor, executing single pass over the network, measuring the time needed
        and finally returning the output.
        :return: dict, output dictionary with tensor names and corresponding output
        """

        def runner_func():
            if self._do_autocast:
                context = torch.cpu.amp.autocast()
            elif self._gpu_autocast:
                context = torch.cuda.amp.autocast()
            else:
                context = nullcontext()

            with context:
                start = time.time()
                output = self._model(*args, **kwargs)
                finish = time.time()
                if self._gpu:
                    torch.cuda.synchronize()
                    finish = time.time()

            self._start_times.append(start)
            self._finish_times.append(finish)
            self.set_task_size(task_size)
            self._times_invoked += 1

            return output

        with torch.no_grad():
            if self._do_profile:
                with profile() as self._profile:
                    return runner_func()
            else:
                return runner_func()

    def print_performance_metrics(self):
        if self._do_profile:
            print(self._profile.key_averages().table(sort_by='cpu_time_total', row_limit=50))
            torch._C._aio_profiler_print()
        return self.print_metrics()


def check_if_cached(model):
    cached_dir = Path(os.path.dirname(os.path.realpath(__file__)), "torch_jit_cache")
    if not cached_dir.exists():
        cached_dir.mkdir()
    cached_path = cached_dir / f"{hashlib.sha224(str(model).encode('utf-8')).hexdigest()}.pt"
    return cached_path.exists(), cached_path


def load_from_cache_or_apply(model, conversion):
    is_cached, cached_path = check_if_cached(model)
    if is_cached:
        print("Loading from cache ...")
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


def apply_jit_trace_module(model, example_inputs):
    return load_from_cache_or_apply(model, lambda: torch.jit.trace_module(model, example_inputs))


def apply_compile(model):
    if os.environ.get("TORCH_COMPILE") == "0":
        return model
    if version.parse(pkg_resources.get_distribution("torch").version) >= version.parse("1.14"):
        # More natural comparison to version.parse("2.0") returns False for 2.0.0a0+git07156c4.dev, which is wrong.
        if '_aio_profiler_print' in dir(torch._C) and os.environ.get("AIO_PROCESS_MODE") != "0":
            backend = "aio"
            options = {"modelname": model.__self__._get_name()
            if isinstance(model, types.MethodType) else model._get_name()}
            utils.print_warning_message(
                f"AIO available and enabled, applying torch.compile() with \"{backend}\" backend.")
        else:
            backend = "inductor"
            options = {}
            utils.print_warning_message(
                f"AIO unavailable or disabled, applying torch.compile() with \"{backend}\" backend.")
        model = torch.compile(model, backend=backend, options=options)
        return model
    else:
        utils.print_goodbye_message_and_die(
            f"Installed PyTorch version is {pkg_resources.get_distribution('torch').version}. "
            f"PyTorch version must be at least 2.0.0 to use torch.compile().")


class SkipScript(Exception):
    pass
