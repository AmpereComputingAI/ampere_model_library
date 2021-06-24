import tensorflow as tf
import tensorflow_hub as hub
import utils.benchmark as bench_utils
import time

class YamnetRunner:
    """
        A class providing facilities to run TensorFlow saved model from tensorflow hub.
    """

    def __init__(self, url_to_model):
        """
        A function initializing runner by providing url to model.

        :param url_to_model: str, eg. "https://tfhub.dev/google/yamnet/1"
        """
        tf.config.threading.set_intra_op_parallelism_threads(bench_utils.get_intra_op_parallelism_threads())
        tf.config.threading.set_inter_op_parallelism_threads(1)
        self.__model = hub.load(url_to_model)
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
        output, embeddings, spectrogram = self.__model(input)
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

    def get_class_map_path(self):
        return self.__model.class_map_path().numpy()
