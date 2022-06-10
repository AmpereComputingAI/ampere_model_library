import os
import csv
import json
import shutil
import argparse
import statistics
import subprocess
from utils.misc import print_goodbye_message_and_die


def parse_args():
    parser = argparse.ArgumentParser(description="Run multi process benchmark.")
    parser.add_argument("-e", "--executable",
                        type=str, required=True,
                        help="path to the python executable to be run")
    parser.add_argument("-a", "--args",
                        type=str, required=True,
                        help="args for executable")
    parser.add_argument("-n", "--num_processes",
                        type=int, default=1,
                        help="number of processes to spawn")
    parser.add_argument("-t", "--num_threads",
                        type=int, default=1,
                        help="number of threads to use per process")
    parser.add_argument("--debug",
                        action="store_true",
                        help="print stdout + stderr of processes?")
    parser.add_argument("--skip_warm_up",
                        action="store_true",
                        help="skip warm-up run?")
    return parser.parse_args()


def gen_threads_config(num_threads, process_id):
    first_thread = process_id * num_threads
    last_thread = (process_id + 1) * num_threads
    threads_to_use = [str(i) for i in range(first_thread, last_thread)]
    return " ".join(threads_to_use), ",".join(threads_to_use)


def calculate_throughput(args, logs_dir, logs):
    with open(os.path.join(logs_dir, f"meta_{logs[0][:-4]}.json"), "r") as meta_file:
        batch_size = json.load(meta_file)["batch_size"]

    runs = dict()
    latest_start = None
    earliest_finish = None
    for log in logs:
        with open(os.path.join(logs_dir, log), "r") as log_file:
            reader = csv.reader(log_file, delimiter=",")
            i = 0
            for row in reader:
                if i == 0:
                    start_time = float(row[0])
                    if latest_start is None or latest_start < start_time:
                        latest_start = start_time
                    runs[log] = [[float(time) for time in row]]
                elif i == 1:
                    finish_time = float(row[-1])
                    if earliest_finish is None or finish_time < earliest_finish:
                        earliest_finish = finish_time
                    runs[log].append([float(time) for time in row])
                else:
                    print_goodbye_message_and_die("Corrupted CSV files with results.")
                i += 1

    latencies = list()
    for pid, run in runs.items():
        observations = len(run[0])
        assert observations == len(run[1])
        inputs_processed = 0
        for i in range(observations):
            start = run[0][i]
            finish = run[1][i]
            if start >= latest_start and finish <= earliest_finish:
                inputs_processed += 1
                latency = finish - start
                latencies.append(latency)
            elif earliest_finish < finish:
                break
        if inputs_processed < 1:
            print_goodbye_message_and_die(
                "Subprocess with pid {pid[:-4]} has not produced any output while running in parallel, "
                "consider increasing timeout.")

    overlap_time = earliest_finish - latest_start
    print("\n Overlap time of processes: {:.2f} s".format(overlap_time))

    batch_x_proc = batch_size * args.num_processes
    throughput_mean = batch_x_proc / statistics.mean(latencies)
    throughput_median = batch_x_proc / statistics.median(latencies)
    print(" Throughput:     mean= {:>10.2f} ips,     median= {:>10.2f} ips\n".format(
        throughput_mean, throughput_median))
    return throughput_mean, throughput_median


def main():
    args = parse_args()
    exec_args = args.args.split()

    if not args.skip_warm_up:
        os.environ["AIO_NUMA_CPUS"] = "1"
        os.environ["DLS_NUMA_CPUS"] = "1"
        os.environ["WARM_UP_ONLY"] = "1"
        cmd = ["python3", args.executable] + exec_args
        if args.debug:
            warm_up = subprocess.Popen(cmd)
        else:
            warm_up = subprocess.Popen(cmd, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb'))
        if warm_up.wait() != 0:
            print_goodbye_message_and_die("Warm-up run died, consider running with --debug")
        os.environ["WARM_UP_ONLY"] = "0"

    os.environ["IGNORE_DATASET_LIMITS"] = "1"

    results_dir = os.path.join(os.getcwd(), "cache")
    os.environ["RESULTS_DIR"] = results_dir
    if os.path.exists(results_dir) and os.path.isdir(results_dir):
        shutil.rmtree(results_dir)
    os.mkdir(results_dir)

    current_subprocesses = list()
    for n in range(args.num_processes):
        aio_numa_cpus, physcpubind = gen_threads_config(args.num_threads, n)
        os.environ["AIO_NUMA_CPUS"] = aio_numa_cpus
        os.environ["DLS_NUMA_CPUS"] = aio_numa_cpus
        cmd = ["numactl", f"--physcpubind={physcpubind}", "python3", args.executable] + exec_args

        if args.debug:
            current_subprocesses.append(subprocess.Popen(cmd))
        else:
            current_subprocesses.append(subprocess.Popen(
                cmd, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb')))

    exit_codes = [p.wait() for p in current_subprocesses]
    if not all(exit_code == 0 for exit_code in exit_codes):
        print(exit_codes)
        print_goodbye_message_and_die("At least one of subprocesses returned exit code 1!")

    logs = [log for log in os.listdir(results_dir) if "csv" in log]
    if len(logs) != args.num_processes:
        print_goodbye_message_and_die("At least one of subprocesses failed to dump results!")

    calculate_throughput(args, results_dir, logs)


if __name__ == "__main__":
    main()
