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
from typing import List

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


class Runner:
    """
    start_times: list, list with timestamps at which inference was called
    finish_times: list, list with timestamps at which results of inference were returned
    times_invoked: int, number of runs completed
    workload_size: List[int], list containing size of each task - size is understood as either user adjustable or input
    dependent component of the input tensor - examples: 1. processing 16 images in a batch, each of fixed dims
    224x224x3, followed by 8 images in a batch of the same fixed shape, workload size = [16, 8]; 2. processing audio of
    length 117 seconds, followed by audio of length 233 seconds, workload size = [117, 233]. I.e. single task size is a
    product of input dimensions' values that can change with next task (task == work to be done during single pass
    through a network)
    warm_up_runs: int, number of warm-up runs to exclude from final metrics
    """

    warm_up_runs = 2

    def __init__(self):
        self._times_invoked = 0
        self._start_times = list()
        self._finish_times = list()
        self._workload_size = list()

    def run(self, task_size: int, *args, **kwargs):
        raise NotImplementedError

    def __print_performance_metrics(self):
        if self._times_invoked == 0:
            utils.print_goodbye_message_and_die(
                "Cannot print performance data as not a single run has been completed! Increase the timeout.")

        if self._times_invoked <= self.warm_up_runs:
            if os.environ.get("IGNORE_PERF_CALC_ERROR") == "1":
                sys.exit(0)
            utils.print_goodbye_message_and_die(
                "Cannot print performance data as only warm-up run(s) have been completed! Increase the timeout.")
        else:
            assert len(self._start_times) == len(self._finish_times) == self._times_invoked == len(self._workload_size)

            latencies = [self._finish_times[i] - self._start_times[i]
                         for i in range(self.warm_up_runs, self._times_invoked)]
            observed_throughput = sum(self._workload_size[self.warm_up_runs:]) / sum(latencies)

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
                "observed_throughput_ips": observed_throughput,
                "inverted_throughput_ms": ms_in_sec / observed_throughput
            }

            max_len = 10
            metrics_lat = {"mean": "mean_lat_ms",
                           "median": "median_lat_ms",
                           "p90": "90th_percentile_lat_ms",
                           "p99": "99th_percentile_lat_ms",
                           "p99.9": "99.9th_percentile_lat_ms"}
            indent = 2 * " "
            print(f"\n{indent}LATENCY")
            for metric in metrics_lat.keys():
                output = f"{3 * indent}{metric}{(max_len - len(metric)) * ' '}{3 * indent}" \
                         + "{:>10.2f} [ms]".format(results[metrics_lat[metric]])
                if os.environ.get("ENABLE_NORMALIZATION") == "1":
                    output += " [NORMALIZED]"
                print(output)

            print(f"\n{indent}THROUGHPUT")
            print(f"{3 * indent}observed{(max_len - len('observed')) * ' '}{3 * indent}"
                  + "{:>10.2f} [samples/s]".format(results["observed_throughput_ips"]))
            print(f"{3 * indent}inverted{(max_len - len('inverted')) * ' '}{3 * indent}"
                  + "{:>10.2f} [ms]".format(results["inverted_throughput_ms"]))

            print(f"\n{indent}Performance results above are based on {len(latencies)} sample(s).")
            print(f"{indent}{self.warm_up_runs} warm-up runs have not been considered.")

            dump_results_maybe(start_times, finish_times, variable_input_sizes, warm_up_runs)
            return results


def run_model(single_pass_func, runner, dataset, batch_size, num_runs, timeout):
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
    :return: dict containing accuracy metrics and dict containing perf metrics
    """
    if num_runs is not None:
        requested_instances_num = num_runs * batch_size
        if dataset.available_instances < requested_instances_num:
            utils.print_goodbye_message_and_die(
                f"Number of runs requested exceeds number of instances available in dataset! "
                f"(Requested: {requested_instances_num}, Available: {dataset.available_instances})")

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

    return dataset.summarize_accuracy(), runner.print_performance_metrics()


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
        print(f"\n  Results have been dumped to {dump_path}\n")
