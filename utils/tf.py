import os
import time
import utils.miscellaneous as utils
import tensorflow.compat.v1 as tf

INTRA_OP_PARALLELISM_THREADS = None


def set_global_intra_op_parallelism_threads(num_intra_threads: int):
    """
    A function setting num of intra threads to be used globally - perfect to use when num of threads is passed through
    argparse to subordinate script (as opposed to setting recognized system environment variable such as
    OMP_NUM_THREADS by user).

    :param num_intra_threads: int, self-explanatory
    """
    global INTRA_OP_PARALLELISM_THREADS
    INTRA_OP_PARALLELISM_THREADS = num_intra_threads


def get_intra_op_parallelism_threads():
    """
    A function checking the value of global variable INTRA_OP_PARALLELISM_THREADS - if unset recognized system
    environment variables are checked. If they are unset as well a fail message is printed and program quits.

    :return: value of global variable INTRA_OP_PARALLELISM_THREADS
    """
    global INTRA_OP_PARALLELISM_THREADS
    if INTRA_OP_PARALLELISM_THREADS is None:
        try:
            INTRA_OP_PARALLELISM_THREADS = int(os.environ["OMP_NUM_THREADS"])
        except KeyError:
            INTRA_OP_PARALLELISM_THREADS = int(os.environ["DLS_NUM_THREADS"])
        finally:
            if INTRA_OP_PARALLELISM_THREADS is None:
                utils.print_goodbye_message_and_die("Number of intra threads to use is not set!")
    return INTRA_OP_PARALLELISM_THREADS


def print_performance_metrics(
        warm_up_run_latency: float, total_inference_time: float, number_of_runs: int, batch_size: int):
    """
    A function printing two performance metrics: latency and throughput.

    :param warm_up_run_latency: float, warm up latency expressed in seconds
    :param total_inference_time: float, total inference time expressed in seconds
    :param number_of_runs: int, number of runs completed
    :param batch_size: int, batch size - if batch size was varying over the runs an average should be supplied
    """
    if number_of_runs == 0:
        utils.print_goodbye_message_and_die("Cannot print performance data as not a single run has been completed!")
        
    if number_of_runs == 1:
        utils.print_warning_message("Printing performance data based just on a single (warm-up) run!")
        latency_in_seconds = warm_up_run_latency
    else:
        latency_in_seconds = (total_inference_time - warm_up_run_latency) / (number_of_runs - 1)

    latency_in_ms = latency_in_seconds * 1000
    instances_per_second = batch_size / latency_in_seconds
    print("\nLatency: {:.0f} ms".format(latency_in_ms))
    print("Throughput: {:.2f} ips".format(instances_per_second))


class TFFrozenModelRunner:
    """
    A class providing facilities to run TensorFlow frozen model (in frozen .pb format).
    """
    def __init__(self, path_to_model: str, output_names: list):
        """
        A function initializing runner by providing path to model and list of output names (can be easily checked with
        Netron app).

        :param path_to_model: str, eg. "ssd_mobilenet_v2_coco_2018_03_29/frozen_inference_graph.pb"
        :param output_names: list of str, eg. ["detection_classes:0", "detection_boxes:0"]
        """
        self.__graph = self.__initialize_graph(path_to_model)
        self.__sess = tf.Session(config=self.__create_config(get_intra_op_parallelism_threads()), graph=self.__graph)
        self.__feed_dict = dict()
        self.__output_dict = {output_name: self.__graph.get_tensor_by_name(output_name) for output_name in output_names}
        self.__warm_up_run_latency = 0.0
        self.__total_inference_time = 0.0
        self.__times_invoked = 0

    def __create_config(self, intra_threads: int, inter_threads=1):
        """
        A function creating TF config for given num of threads.

        :param intra_threads: int
        :param inter_threads: int
        :return: TensorFlow config
        """
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.intra_op_parallelism_threads = intra_threads
        config.inter_op_parallelism_threads = inter_threads
        return config

    def __initialize_graph(self, path_to_model: str):
        """
        A function initializing TF graph from frozen .pb model.

        :param path_to_model: str
        :return: TensorFlow graph
        """
        graph = tf.Graph()
        with graph.as_default():
            graph_def = tf.GraphDef()
            with tf.gfile.GFile(path_to_model, 'rb') as fid:
                serialized_graph = fid.read()
                graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(graph_def, name="")
        return graph

    def set_input_tensor(self, input_name: str, input_array):
        """
        A function assigning given numpy input array to the tensor under the provided input name.

        :param input_name: str, name of a input node in a model, eg. "image_tensor:0"
        :param input_array: numpy array with intended input
        """
        self.__feed_dict[self.__graph.get_tensor_by_name(input_name)] = input_array

    def run(self):
        """
        A function executing single pass over the network, measuring the time need and returning the output.

        :return: dict, output dictionary with tensor names and corresponding output
        """
        start = time.time()
        output = self.__sess.run(self.__output_dict, self.__feed_dict)
        finish = time.time()
        self.__total_inference_time += finish - start
        if self.__times_invoked == 0:
            self.__warm_up_run_latency += finish - start
        self.__times_invoked += 1
        return output

    def print_performance_metrics(self, batch_size):
        """
        A function printing performance metrics on runs executed by the runner so far and then closing TF session.

        :param batch_size: int, batch size - if batch size was varying over the runs an average should be supplied
        """
        print_performance_metrics(
            self.__warm_up_run_latency, self.__total_inference_time, self.__times_invoked, batch_size)
        self.__sess.close()
