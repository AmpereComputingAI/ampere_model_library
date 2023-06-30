# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC
import csv
import json
import os
import sys
import time
import statistics
import numpy as np
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


def run_model(single_pass_func, runner, dataset, batch_size, num_runs, timeout,
              variable_input_lengths=None):
    """
    A function running model in unified way.

    If num_runs is specified the function will execute single_pass_func n times and then summarize accuracy and perf.
    If num_runs is unspecified (None) the function will execute single_pass_func until either timeout is reached or
    end of dataset.

    :param single_pass_func: python function that:
        1. pre_processes input, sets input tensor
        2. invokes the run by a call to runner
        3. post-processes the output
    :param runner: python class providing the unified runner facilities
    :param dataset: python class providing the unified dataset facilities
    :param batch_size: int, batch size
    :param num_runs: int, number of times that single_pass_func should be executed
    :param timeout: float, time in seconds after which iterations over single_pass_func should be stopped
    :param variable_input_lengths: list[int], variable lengths of input tensors in the order of execution
    :return: dict containing accuracy metrics and dict containing perf metrics
    """
    if num_runs is not None:
        requested_instances_num = num_runs * batch_size
        if dataset.available_instances < requested_instances_num:
            utils.print_goodbye_message_and_die(
                f"Number of runs requested exceeds number of instances available in dataset! "
                f"(Requested: {requested_instances_num}, Available: {dataset.available_instances})")

    if variable_input_lengths is not None:
        utils.print_warning_message(
            "Input has variable shape, it is recommended to run this benchmark for a fixed number of runs")

    if os.environ.get("WARM_UP_ONLY") == "1":
        single_pass_func(runner, dataset)
        sys.exit(0)

    if num_runs is None:
        timeout_pbar = tqdm(total=int(timeout))
    start = time.time()
    while True:
        try:
            if num_runs is None:
                while time.time() - start < timeout:
                    single_pass_func(runner, dataset)
                    timeout_pbar.n = int(min(time.time() - start, timeout))
                    timeout_pbar.refresh()
            else:
                for _ in tqdm(range(num_runs)):
                    single_pass_func(runner, dataset)
        except utils.OutOfInstances:
            if os.environ.get("IGNORE_DATASET_LIMITS") == "1":
                assert num_runs is None, "IGNORE_DATASET_LIMITS=1 can't be set for defined number of runs"
                if dataset.reset():
                    continue
        break
    if num_runs is None:
        timeout_pbar.close()

    return dataset.summarize_accuracy(), runner.print_performance_metrics(batch_size, variable_input_lengths)


def print_performance_metrics(
        start_times: list, finish_times: list, num_runs: int, batch_size: int, warm_up_runs=2,
        variable_input_lengths=None):
    """
    A function printing two performance metrics: latency and throughput.

    :param start_times: list, list with timestamps at which inference was called
    :param finish_times: list, list with timestamps at which results of inference were returned
    :param num_runs: int, number of runs completed
    :param batch_size: int, batch size - if batch size was varying over the runs an average should be supplied
    :param warm_up_runs: int, number of warm-up runs to exclude from final metrics
    :param variable_input_lengths: list[int], variable lengths of input tensors in the order of execution
    """
    if num_runs == 0:
        utils.print_goodbye_message_and_die(
            "Cannot print performance data as not a single run has been completed! Increase the timeout.")

    if num_runs <= warm_up_runs:
        if os.environ.get("IGNORE_PERF_CALC_ERROR") == "1":
            sys.exit(0)
        utils.print_goodbye_message_and_die(
            "Cannot print performance data as only warm-up run(s) have been completed! Increase the timeout.")
    else:
        assert len(start_times) == len(finish_times) == num_runs

        variable_input_lengths_sum = num_runs - warm_up_runs
        if variable_input_lengths is not None:
            min_input_length, max_input_length = \
                min(variable_input_lengths[warm_up_runs:]), max(variable_input_lengths[warm_up_runs:])
            utils.print_warning_message(
                f"Latency results will be normalized due to variable input shape "
                f"(min = {min_input_length} ; max = {max_input_length}), to disable normalization run with "
                f"DISABLE_NORMALIZATION=1")
            assert num_runs == len(variable_input_lengths)
            average_input_length = statistics.mean(variable_input_lengths)
            input_length_factors = [input_length / average_input_length for input_length in variable_input_lengths]
            variable_input_lengths_sum = sum(variable_input_lengths[warm_up_runs:])

        latencies = []
        for i in range(warm_up_runs, num_runs):
            latency_sec = finish_times[i] - start_times[i]
            if variable_input_lengths is not None and os.environ.get("DISABLE_NORMALIZATION") != "1":
                latency_sec /= input_length_factors[i]
            latencies.append(latency_sec)

        mean_latency_sec = statistics.mean(latencies)
        median_latency_sec = statistics.median(latencies)
        percentile_90th_latency_sec = np.percentile(latencies, 90)
        percentile_99th_latency_sec = np.percentile(latencies, 99)
        percentile_999th_latency_sec = np.percentile(latencies, 99.9)

        ms_in_sec = 1000
        results = {
            "mean_lat_ms": mean_latency_sec * ms_in_sec,
            "median_lat_ms": median_latency_sec * ms_in_sec,
            "90th_percentile_lat_ms": percentile_90th_latency_sec * ms_in_sec,
            "99th_percentile_lat_ms": percentile_99th_latency_sec * ms_in_sec,
            "99.9th_percentile_lat_ms": percentile_999th_latency_sec * ms_in_sec,
            "observed_throughput": batch_size * variable_input_lengths_sum / sum(latencies)
        }

        metrics_lat = {"mean": "mean_lat_ms",
                       "median": "median_lat_ms",
                       "p90": "90th_percentile_lat_ms",
                       "p99": "99th_percentile_lat_ms",
                       "p99.9": "99.9th_percentile_lat_ms"}
        max_len = max([len(metric) for metric in metrics_lat.keys()])
        indent = 2 * " "
        print(f"\n{indent}LATENCY")
        for metric in metrics_lat.keys():
            print(f"{3 * indent}{metric}{(max_len - len(metric)) * ' '}{3 * indent}"
                  + "{:>10.2f} [ms]".format(results[metrics_lat[metric]]))

        metrics_throughput = {"observed": "observed_throughput"}
        print(f"\n{indent}THROUGHPUT")
        for metric in metrics_throughput.keys():
            print(f"{3 * indent}{metric}{(max_len - len(metric)) * ' '}{3 * indent}"
                  + "{:>10.2f} [samples/s]".format(results[metrics_throughput[metric]]))

        print(f"\n{indent}Performance results above are based on {len(latencies)} sample(s).")
        print(f"{indent}{warm_up_runs} warm-up runs have not been considered.\n")

        if variable_input_lengths is None:
            variable_input_sizes = [batch_size for _ in range(num_runs)]
        else:
            variable_input_sizes = [batch_size * variable_input_lengths[i] for i in range(num_runs)]

        dump_csv_results_maybe(start_times, finish_times, variable_input_sizes, warm_up_runs)
        return results


def dump_csv_results_maybe(start_times, finish_times, variable_input_sizes, warm_up_runs=2):
    # variable input sizes mean the size of input tensors along dimensions that are configurable by user in the order
    # of execution, e.g. input sizes for 2 runs of NLP model with bs=8 and seq_sizes of [384, 512] across two subsequent
    # runs would be [8*384, 8*512]
    dump_dir = os.environ.get("RESULTS_DIR")
    if dump_dir is not None and len(start_times) > warm_up_runs:
        dump_path = os.path.join(dump_dir, f"{os.getpid()}.json")
        with open(dump_path, "w") as f:
            json.dump({
                "input_sizes": variable_input_sizes[warm_up_runs:],
                "start_times": start_times[warm_up_runs:],
                "finish_times": finish_times[warm_up_runs:]
            }, f)
        print(f"  Results have been dumped to {dump_path}\n")
