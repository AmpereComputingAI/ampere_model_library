# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import os
import time
import onnxruntime as ort
from utils.benchmark import get_intra_op_parallelism_threads, Runner
from utils.misc import advertise_aio, check_memory_settings


class OrtRunner(Runner):
    """
    A class providing facilities to run ONNX model
    """

    def __init__(self, model: str, throughput_only=False):
        super().__init__(throughput_only)
        try:
            ort.AIO
        except AttributeError:
            advertise_aio("ONNXRunTime")
        check_memory_settings()

        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = get_intra_op_parallelism_threads()
        if os.getenv("ORT_PROFILER", "0") == "1":
            session_options.enable_profiling = True

        self.session = ort.InferenceSession(model, session_options, providers=ort.get_available_providers())

        self._feed_dict = dict()
        self._output_names = [output.name for output in self.session.get_outputs()]

        print("\nRunning with ONNX Runtime\n")

    def run(self, task_size=None, *args, **kwargs):
        start = time.time()
        outputs = self.session.run(self._output_names, self._feed_dict)
        finish = time.time()

        self._start_times.append(start)
        self._finish_times.append(finish)
        self.set_task_size(task_size)
        self._times_invoked += 1

        return outputs

    def set_input_tensor(self, input_name: str, input_array):
        self._feed_dict[input_name] = input_array

    def print_performance_metrics(self):
        """
        A function printing performance metrics on runs executed by the runner so far.
        """
        if os.getenv("AIO_PROFILER", "0") == "1":
            ort.AIO.print_profile_data()
        if os.getenv("ORT_PROFILER", "0") == "1":
            prof = self.session.end_profiling()
            if not os.path.exists("profiler_output/ort/"):
                os.makedirs("profiler_output/ort/")
            os.replace(prof, f"profiler_output/ort/{prof}")

        return self.print_metrics()
