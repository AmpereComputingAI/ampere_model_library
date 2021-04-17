import os
import utils.misc as utils

intra_op_parallelism_threads = None


def set_global_intra_op_parallelism_threads(num_intra_threads: int):
    """
    A function setting num of intra threads to be used globally - perfect to use when num of threads is passed through
    argparse to subordinate script (as opposed to setting recognized system environment variable such as
    OMP_NUM_THREADS by user).

    :param num_intra_threads: int, self-explanatory
    """
    global intra_op_parallelism_threads
    intra_op_parallelism_threads = num_intra_threads


def get_intra_op_parallelism_threads():
    """
    A function checking the value of global variable intra_op_parallelism_threads - if unset recognized system
    environment variables are checked. If they are unset as well a fail message is printed and program quits.

    :return: value of global variable intra_op_parallelism_threads
    """
    global intra_op_parallelism_threads
    if intra_op_parallelism_threads is None:

        try:
            omp_num_threads = int(os.environ["OMP_NUM_THREADS"])
            intra_op_parallelism_threads = omp_num_threads
        except KeyError:
            omp_num_threads = None

        try:
            dls_num_threads = int(os.environ["DLS_NUM_THREADS"])
            if omp_num_threads is not None:
                if omp_num_threads != dls_num_threads:
                    utils.print_goodbye_message_and_die(
                        f"DLS_NUM_THREADS={dls_num_threads} inconsistent with OMP_NUM_THREADS={omp_num_threads}!")
            intra_op_parallelism_threads = dls_num_threads
        except KeyError:
            pass

        if intra_op_parallelism_threads is None:
            utils.print_goodbye_message_and_die("Number of intra threads to use is not set!"
                                                "\nUse DLS_NUM_THREADS or OMP_NUM_THREADS flags.")

        print(f"\nRunning with {intra_op_parallelism_threads} threads\n")

    return intra_op_parallelism_threads


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
