# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import os
import sys
import time
import statistics
import utils.misc as utils
from tqdm.auto import tqdm

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
            aio_num_threads = int(os.environ["AIO_NUM_THREADS"])
            if omp_num_threads is not None:
                if omp_num_threads != aio_num_threads:
                    utils.print_goodbye_message_and_die(
                        f"AIO_NUM_THREADS={aio_num_threads} inconsistent with OMP_NUM_THREADS={omp_num_threads}!")
            intra_op_parallelism_threads = aio_num_threads
        except KeyError:
            pass

        if intra_op_parallelism_threads is None:
            utils.print_goodbye_message_and_die("Number of intra threads to use is not set!"
                                                "\nUse AIO_NUM_THREADS or OMP_NUM_THREADS flags.")

        print(f"\nIntraop parallelism set to {intra_op_parallelism_threads} threads\n")

    return intra_op_parallelism_threads


def benchmark_func(func, num_runs, timeout, warm_up=True):
    """
    A function for benchmarking functions compliant to model_zoo approach in other parts of the code.

    :param func: python function to be benchmarked
    :param num_runs: int, number of func invocations to be done
    :param timeout: float, time expressed in seconds after which benchmarking should be stopped
    :param warm_up: bool, whether to do a single warm-up run excluded from measurements

    :return: latency in seconds
    """
    def benchmark(function):
        start = time.time()
        function()
        return time.time() - start

    if warm_up:
        _ = benchmark(func)

    latencies = list()
    if num_runs is None:
        i = 0
        benchmarking_start = time.time()
        while time.time() - benchmarking_start < timeout:
            latencies.append(benchmark(func))
            i += 1
    else:
        i = num_runs
        for _ in tqdm(range(num_runs)):
            latencies.append(benchmark(func))

    return sum(latencies) / i


def run_model(single_pass_func, runner, dataset, batch_size, num_runs, timeout):
    """
    A function running model in unified way.

    If num_runs is specified the function will execute single_pass_func n times and then summarize accuracy and perf.
    If num_runs is unspecified (None) the function will execute single_pass_func until either timeout is reached or
    end of dataset.

    :param single_pass_func: python function that:
        1. sets input tensor,
        2. invokes the run by a call to runner,
        3. post-processes the output
    :param runner: python class providing the unified runner facilities
    :param dataset: python class providing the unified dataset facilities
    :param batch_size: int, batch size
    :param num_runs: int, number of times that single_pass_func should be executed
    :param timeout: float, time in seconds after which iterations over single_pass_func should be stopped
    :return: dict containing accuracy metrics and dict containing perf metrics
    """
    if num_runs is not None:
        requested_instances_num = num_runs * batch_size
        if dataset.available_instances < requested_instances_num:
            utils.print_goodbye_message_and_die(
                f"Number of runs requested exceeds number of instances available in dataset! "
                f"(Requested: {requested_instances_num}, Available: {dataset.available_instances})")

    start = time.time()
    try:
        if os.environ.get("WARM_UP_ONLY") == "1":
            single_pass_func(runner, dataset)
            sys.exit(0)

        if num_runs is None:
            single_pass_func(runner, dataset)
            while time.time() - start < timeout:
                single_pass_func(runner, dataset)
        else:
            for _ in tqdm(range(num_runs)):
                single_pass_func(runner, dataset)
    except utils.OutOfInstances:
        if os.environ.get("IGNORE_DATASET_LIMITS") == "1" and num_runs is None:
            if dataset.reset():
                return run_model(
                    single_pass_func, runner, dataset, batch_size, num_runs, timeout - (time.time() - start))

    return dataset.summarize_accuracy(), runner.print_performance_metrics(batch_size)


def print_performance_metrics(start_times: list, finish_times: list, num_runs: int, batch_size: int, warm_up_runs=2):
    """
    A function printing two performance metrics: latency and throughput.

    :param start_times: list, list with timestamps at which inference was called
    :param finish_times: list, list with timestamps at which results of inference were returned
    :param num_runs: int, number of runs completed
    :param batch_size: int, batch size - if batch size was varying over the runs an average should be supplied
    :param warm_up_runs: int, number of warm-up runs to exclude from final metrics
    """
    if num_runs == 0:
        utils.print_goodbye_message_and_die(
            "Cannot print performance data as not a single run has been completed! Increase the timeout.")

    if num_runs <= warm_up_runs:
        utils.print_goodbye_message_and_die(
            "Cannot print performance data as only warm-up run(s) have been completed! Increase the timeout.")
    else:
        assert len(start_times) == len(finish_times) == num_runs

        latencies = []
        for i in range(warm_up_runs, num_runs):
            latencies.append(finish_times[i] - start_times[i])

        mean_latency_sec = statistics.mean(latencies)
        median_latency_sec = statistics.median(latencies)

        mean_latency_ms = mean_latency_sec * 1000
        median_latency_ms = median_latency_sec * 1000
        mean_throughput = batch_size / mean_latency_sec
        median_throughput = batch_size / median_latency_sec

        print("\n Latency:        mean= {:>10.0f}  ms,     median= {:>10.0f}  ms".format(
            mean_latency_ms, median_latency_ms))
        print(" Throughput:     mean= {:>10.2f} ips,     median= {:>10.2f} ips\n".format(
            mean_throughput, median_throughput))
        return {"mean_lat_ms": mean_latency_ms, "median_lat_ms": median_latency_ms,
                "mean_throughput": mean_throughput, "median_throughput": median_throughput}
