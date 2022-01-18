import os
import onnxruntime as ort
import time
import utils.benchmark as bench_utils

class OrtRunner:
    """
    A class providing facilities to run ONNX model
    """

    def __init__(self, model: str):
        session_options = ort.SessionOptions()
        session_options.intra_op_num_threads = bench_utils.get_intra_op_parallelism_threads()
        if os.getenv("ORT_PROFILER", "0") == "1":
            session_options.enable_profiling = True

        self.session = ort.InferenceSession(model, session_options)

        self.__feed_dict = dict()
        self.__output_names = [output.name for output in self.session.get_outputs()]

        self.__warm_up_run_latency = 0.0
        self.__total_inference_time = 0.0
        self.__times_invoked = 0

        print("\nRunning with ONNX Runtime\n")

    def run(self):

        start = time.time()
        outputs = self.session.run(self.__output_names, self.__feed_dict)
        finish = time.time()

        self.__total_inference_time += finish - start
        if self.__times_invoked == 0:
                self.__warm_up_run_latency += finish - start
        self.__times_invoked += 1

        return outputs
    
    def set_input_tensor(self, input_name: str, input_array):
        self.__feed_dict[input_name] = input_array

    def print_performance_metrics(self, batch_size):
        """
        A function printing performance metrics on runs executed by the runner so far.
        :param batch_size: int, batch size - if batch size was varying over the runs an average should be supplied
        """
        perf = bench_utils.print_performance_metrics(
            self.__warm_up_run_latency, self.__total_inference_time, self.__times_invoked, batch_size)
        if os.getenv("AIO_PROFILER", "0") == "1":
            ort.AIO.print_profile_data()
        if os.getenv("ORT_PROFILER", "0") == "1":
            prof = self.session.end_profiling()
            if not os.path.exists("profiler_output/ort/"):
                os.makedirs("profiler_output/ort/")
            os.replace(prof, f"profiler_output/ort/{prof}")
        return perf