# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import os
import sys
import json
import time
import argparse
import subprocess
import urllib.request
from pathlib import Path

LATEST_VERSION = "2.2.0a0+git1bef725"
SYSTEMS = {
    "Altra": {
        "ResNet-50 v1.5": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/q80_30%40ampere_pytorch_1.10.0%40resnet_50_v1.5.json",  # noqa
        "YOLO v8s": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/q80_30%40ampere_pytorch_1.10.0%40yolo_v8_s.json",  # noqa
        "BERT large": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/q80_30%40ampere_pytorch_1.10.0%40bert_large_mlperf_squad.json",  # noqa
        "DLRM": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/q80_30%40ampere_pytorch_1.10.0%40dlrm_torchbench.json",  # noqa
        "Whisper medium EN": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/q80_30%40ampere_pytorch_1.10.0%40whisper_medium.en.json"  # noqa
    },
    "Altra Max": {
        "ResNet-50 v1.5": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/m128_30%40ampere_pytorch_1.10.0%40resnet_50_v1.5.json",  # noqa
        "YOLO v8s": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/m128_30%40ampere_pytorch_1.10.0%40yolo_v8_s.json",  # noqa
        "BERT large": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/m128_30%40ampere_pytorch_1.10.0%40bert_large_mlperf_squad.json",  # noqa
        "DLRM": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/m128_30%40ampere_pytorch_1.10.0%40dlrm_torchbench.json",  # noqa
        "Whisper medium EN": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/m128_30%40ampere_pytorch_1.10.0%40whisper_medium.en.json"  # noqa
    },
    "AmpereOne": {
        "ResNet-50 v1.5": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/siryn%40ampere_pytorch_1.10.0%40resnet_50_v1.5.json",  # noqa
        "YOLO v8s": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/siryn%40ampere_pytorch_1.10.0%40yolo_v8_s.json",  # noqa
        "BERT large": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/siryn%40ampere_pytorch_1.10.0%40bert_large_mlperf_squad.json",  # noqa
        "DLRM": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/siryn%40ampere_pytorch_1.10.0%40dlrm_torchbench.json",  # noqa
        "Whisper medium EN": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/siryn%40ampere_pytorch_1.10.0%40whisper_medium.en.json"  # noqa
    },
    "AmpereOneX": {
        "ResNet-50 v1.5": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/banshee%40ampere_pytorch_1.10.0%40resnet_50_v1.5.json",  # noqa
        "YOLO v8s": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/banshee%40ampere_pytorch_1.10.0%40yolo_v8_s.json",  # noqa
        "BERT large": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/banshee%40ampere_pytorch_1.10.0%40bert_large_mlperf_squad.json",  # noqa
        "DLRM": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/banshee%40ampere_pytorch_1.10.0%40dlrm_torchbench.json",  # noqa
        "Whisper medium EN": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/banshee%40ampere_pytorch_1.10.0%40whisper_medium.en.json"  # noqa
    },
}

AFFIRMATIVE = ["y", "Y", "yes", "YES"]
NEGATIVE = ["n", "N", "no", "NO"]

MAX_DEVIATION = 0.01  # max deviation [abs((value_n+1 / value_n) - 1.)] between sample n and sample n+1
MIN_MEASUREMENTS_IN_OVERLAP_COUNT = 10
DAY_IN_SEC = 60 * 60 * 24
TIMEOUT = 30 * 60
INDENT = 3 * " "
no_interactive = None

os.environ["AIO_SKIP_MASTER_THREAD"] = "1"


def print_maybe(text):
    if not no_interactive:
        print(text)


def print_red(message):
    print(f"\033[91m{message}\033[0m")


def print_green(message):
    print(f"\033[92m{message}\033[0m")


def go_ampere_message():
    print_red(f"\nThis script requires latest Ampere optimized PyTorch ({LATEST_VERSION}) to be installed. "
              "\nConsider using our Docker images available at https://hub.docker.com/u/amperecomputingai")
    sys.exit(1)


def do_the_setup_message():
    print_red("\nBefore running this script please run `bash setup_deb.sh && source set_env_variables.sh`")
    sys.exit(1)


def is_setup_done():
    try:
        import torch
    except ImportError:
        go_ampere_message()
    if '_aio_profiler_print' not in dir(torch._C) or torch.__version__ != LATEST_VERSION:
        go_ampere_message()
    setup_confirmation = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".setup_completed")
    if not os.path.exists(setup_confirmation):
        do_the_setup_message()
    elif open(setup_confirmation, "r").read() != open("/etc/machine-id", "r").read():
        do_the_setup_message()
    if os.environ.get("PYTHONPATH") != os.path.dirname(os.path.realpath(__file__)):
        do_the_setup_message()
    print_green("Setup verified. You are good to go! 🔥")


def press_enter_to_continue():
    if no_interactive:
        return
    input("Press ENTER to continue")
    print()


def get_bool_answer(question):
    if no_interactive:
        return True
    answer = None
    while answer not in AFFIRMATIVE + NEGATIVE:
        answer = input(f"{question} (y/n)").strip()
    print()
    return answer in AFFIRMATIVE


def which_ampere_cpu(flags, num_threads_per_node):
    altra_flags = ['aes', 'asimd', 'asimddp', 'asimdhp', 'asimdrdm', 'atomics', 'cpuid', 'crc32', 'dcpop', 'evtstrm',
                   'fp', 'fphp', 'lrcpc', 'pmull', 'sha1', 'sha2', 'ssbs']
    aone_flags = ['aes', 'asimd', 'asimddp', 'asimdfhm', 'asimdhp', 'asimdrdm', 'atomics', 'bf16', 'bti', 'cpuid',
                  'crc32', 'dcpodp', 'dcpop', 'dit', 'ecv', 'evtstrm', 'fcma', 'flagm', 'flagm2', 'fp', 'fphp', 'frint',
                  'i8mm', 'ilrcpc', 'jscvt', 'lrcpc', 'paca', 'pacg', 'pmull', 'rng', 'sb', 'sha1', 'sha2', 'sha3',
                  'sha512', 'ssbs', 'uscat']
    aonex_flags = ['aes', 'asimd', 'asimddp', 'asimdfhm', 'asimdhp', 'asimdrdm', 'atomics', 'bf16', 'bti', 'cpuid',
                   'crc32', 'dcpodp', 'dcpop', 'dit', 'ecv', 'evtstrm', 'fcma', 'flagm', 'flagm2', 'fp', 'fphp',
                   'frint', 'i8mm', 'ilrcpc', 'jscvt', 'lrcpc', 'paca', 'pacg', 'pmull', 'rng', 'sb', 'sha1', 'sha2',
                   'sha3', 'sha512', 'sm3', 'sm4', 'ssbs', 'uscat']

    if set(altra_flags) == set(flags):
        if num_threads_per_node > 80:
            system = "Altra Max"
        else:
            system = "Altra"
    elif set(aone_flags) == set(flags):
        system = "AmpereOne"
    elif set(aonex_flags) == set(flags):
        system = "AmpereOneX"
    else:
        system = None

    return system


def identify_system(args):
    import psutil
    import subprocess
    from cpuinfo import get_cpu_info

    cpu_info = get_cpu_info()
    flags = cpu_info["flags"]
    num_threads = cpu_info["count"]
    try:
        numa_nodes = int([n for n in subprocess.check_output(["lscpu"]).decode().split("\n")
                          if "NUMA node(s):" in n][0].split()[2])
    except (ValueError, IndexError):
        numa_nodes = 1
    num_threads_per_node = num_threads // numa_nodes
    mem = psutil.virtual_memory()
    memory_total = mem.total / 1024 ** 3
    memory_available = mem.available / 1024 ** 3
    if args.memory is not None:
        memory_available = min(args.memory, memory_available)

    if args.system is None:
        system = which_ampere_cpu(flags, num_threads_per_node)
    else:
        for s in SYSTEMS.keys():
            if args.system == convert_name(s):
                system = s
                break
        else:
            assert False

    def system_identified_as():
        print(f"\nSystem identified as {system}\n[out of {', '.join(SYSTEMS.keys())}]")
        print(f"\nNUMA nodes: {numa_nodes}\nThreads: {num_threads_per_node}\nMemory: {round(memory_total, 2)} [GiB]\n")

    run_selection = True
    if system is not None:
        system_identified_as()
        run_selection = not get_bool_answer("Is this correct?")
    else:
        print_red("\nCouldn't identify system. Are you running this on Ampere CPU?")

    if run_selection:
        print("\nPlease select your system from the following list:")
        system_map = {}
        for i, system in enumerate(SYSTEMS.keys()):
            system_map[i] = system
            print(f"{'' * 3}{i}: {system}")
        print()
        answer = None
        while answer not in system_map.keys():
            try:
                answer = int(input(f"Input number for your system [0-{len(system_map) - 1}]"))
            except ValueError:
                pass
        system = system_map[answer]
        system_identified_as()

    if args.max_threads is not None:
        num_threads_per_node = min(num_threads_per_node, args.max_threads)

    return system, numa_nodes, num_threads_per_node, memory_available


def get_thread_configs(numa_nodes, node_idx, num_threads_node, num_proc, num_threads_per_proc):
    from cpuinfo import get_cpu_info
    num_threads = get_cpu_info()["count"] // numa_nodes
    assert num_threads >= num_threads_node
    assert num_threads_per_proc * num_proc <= num_threads_node

    thread_configs = []
    start_idx = node_idx * num_threads
    end_idx = start_idx + num_threads_per_proc
    for n in range(num_proc):
        threads_to_use = [str(t) for t in range(start_idx, end_idx)]
        assert len(threads_to_use) == num_threads_per_proc
        thread_configs.append((" ".join(threads_to_use), ",".join(threads_to_use)))
        start_idx += num_threads_per_proc
        end_idx += num_threads_per_proc
    return thread_configs


def clean_line():
    print("\r" + 80 * " " + "\r", end='')


def ask_for_patience(text):
    clean_line()
    print(f"\r{INDENT}{text}, stay put 🙏 ...", end='')


class Results:
    def __init__(self, results_dir, processes_count):
        self._results_dir = results_dir
        self._prev_measurements_count = None
        self._prev_throughput_total = None
        self._processes_count = processes_count
        self.stable = False

    def calculate_throughput(self, final_calc=False):
        import psutil
        from filelock import FileLock
        logs = [log for log in os.listdir(self._results_dir) if "json" in log and "lock" not in log]
        if len(logs) != self._processes_count:
            ask_for_patience("benchmark starting, CPU util: {:>3.0f}%".format(psutil.cpu_percent(1)))
            return None

        loaded_logs = []
        for log in logs:
            log_filepath = os.path.join(self._results_dir, log)
            with FileLock(f"{log_filepath}.lock", timeout=60):
                with open(log_filepath, "r") as f:
                    loaded_logs.append(json.load(f))

        measurements_counts = [(len(log["start_times"]), len(log["finish_times"]), len(log["workload_size"])) for log in
                               loaded_logs]
        if not all(x[0] == x[1] == x[2] and x[0] >= MIN_MEASUREMENTS_IN_OVERLAP_COUNT for x in measurements_counts):
            ask_for_patience("benchmark on-going, CPU util: {:>3.0f}%".format(psutil.cpu_percent(1)))
            return None
        latest_start = max(log["start_times"][0] for log in loaded_logs)
        earliest_finish = min(log["finish_times"][-1] for log in loaded_logs)

        measurements_completed_in_overlap_total = 0
        throughput_total = 0.
        for log in loaded_logs:
            input_size_processed_per_process = 0
            total_latency_per_process = 0.
            measurements_completed_in_overlap = 0
            for i in range(len(log["start_times"])):
                start = log["start_times"][i]
                finish = log["finish_times"][i]
                if start >= latest_start and finish <= earliest_finish:
                    input_size_processed_per_process += log["workload_size"][i]
                    total_latency_per_process += finish - start
                    measurements_completed_in_overlap += 1
                elif earliest_finish < finish:
                    break
            if measurements_completed_in_overlap < MIN_MEASUREMENTS_IN_OVERLAP_COUNT:
                ask_for_patience("benchmark on-going, CPU util: {:>3.0f}%".format(psutil.cpu_percent(1)))
                return None
            measurements_completed_in_overlap_total += measurements_completed_in_overlap
            throughput_total += input_size_processed_per_process / total_latency_per_process

        if self._prev_measurements_count is not None and \
                measurements_completed_in_overlap_total > self._prev_measurements_count:
            self.stable = abs((throughput_total / self._prev_throughput_total) - 1.) <= MAX_DEVIATION
        self._prev_throughput_total = throughput_total
        self._prev_measurements_count = measurements_completed_in_overlap_total

        if not self.stable and not final_calc:
            print("\r{}total throughput: {:.2f} ips, CPU util: {:>3.0f}%, stabilizing result ...".format(
                INDENT, throughput_total, psutil.cpu_percent(1)), end='')

        return throughput_total


def run_benchmark(model_script, numa_nodes, num_threads_node, num_proc_node, num_threads_per_proc, start_delay=0):
    assert start_delay >= 0
    if num_proc_node * numa_nodes == 1:
        start_delay = 0

    os.environ["IGNORE_DATASET_LIMITS"] = "1"

    os.environ["AIO_NUM_THREADS"] = str(num_threads_per_proc)
    os.environ["OMP_NUM_THREADS"] = str(num_threads_per_proc)

    results_dir = os.path.join(os.getcwd(), ".cache_aml")
    os.environ["RESULTS_DIR"] = results_dir
    if os.path.exists(results_dir) and os.path.isdir(results_dir):
        for filepath in os.listdir(results_dir):
            os.remove(os.path.join(results_dir, filepath))
    else:
        os.mkdir(results_dir)

    mem_bind = []
    results = Results(results_dir, num_proc_node * numa_nodes)
    current_subprocesses = list()
    failure = False
    for i in range(numa_nodes):
        if failure:
            break
        thread_configs = get_thread_configs(numa_nodes, i, num_threads_node, num_proc_node, num_threads_per_proc)
        if numa_nodes > 1:
            mem_bind = [f"--membind={i}"]
        for n in range(num_proc_node):
            if failure:
                break
            aio_numa_cpus, physcpubind = thread_configs[n]
            os.environ["AIO_NUMA_CPUS"] = aio_numa_cpus
            cmd = ["numactl", f"--physcpubind={physcpubind}"] + mem_bind + ["python3"] + model_script.split()
            log_filename = f"/tmp/aml_log_{i * num_proc_node + n}"
            current_subprocesses.append(subprocess.Popen(
                cmd, stdout=open(log_filename, 'wb'), stderr=open(log_filename, 'wb')))
            if start_delay > 0:
                ask_for_patience("benchmark starting, {:>3} / {} streams online".format(
                    i * num_proc_node + n + 1, num_proc_node * numa_nodes))
                time.sleep(start_delay)
                failure = any(p.poll() is not None and p.poll() != 0 for p in current_subprocesses)

    start = time.time()
    while not all(p.poll() is not None for p in current_subprocesses) and not failure:
        time.sleep(5)
        results.calculate_throughput()
        if time.time() - start < TIMEOUT:
            failure = any(p.poll() is not None and p.poll() != 0 for p in current_subprocesses)
        else:
            failure = True
        if results.stable or failure:
            Path(os.path.join(results_dir, "STOP")).touch()
            break

    if not failure:
        # wait for subprocesses to finish their job if all are alive till now
        ask_for_patience("benchmark finishing")
        try:
            failure = any(
                p.wait(timeout=max(0, int(TIMEOUT - (time.time() - start)))) != 0 for p in current_subprocesses)
        except subprocess.TimeoutExpired:
            failure = True

    clean_line()
    if failure:
        exit_codes = []
        for p in current_subprocesses:
            exit_codes.append(p.poll())
            p.terminate()
        if any([code is not None and code != 0 for code in exit_codes]):
            print_red("\nFAIL: At least one process returned exit code other than 0")
            if get_bool_answer("Do you want to print output of failed processes?"):
                for i, p in enumerate(current_subprocesses):
                    if exit_codes[i] != 0 and exit_codes[i] is not None:
                        print_red(f"\nOutput of process {i}:")
                        print(open(f"/tmp/aml_log_{i}", "r", encoding="utf8", errors="ignore").read())
        else:
            print_red(f"\nFAIL: Timeout hit [{TIMEOUT} sec]")
        if not get_bool_answer("Do you want to continue evaluation?"):
            sys.exit(0)
        return None
    else:
        return results.calculate_throughput(final_calc=True)


class Runner:
    def __init__(self, system, model_name, numa_nodes, num_threads, memory, precisions):
        self.model_name = model_name
        self.numa_nodes = numa_nodes
        self.num_threads = num_threads
        self.precisions = precisions
        with urllib.request.urlopen(SYSTEMS[system][model_name]) as url:
            look_up_data = json.load(url)
        from utils.perf_prediction.predictor import find_best_config
        self._results = {precision: {} for precision in precisions}
        print_maybe("Expected performance on your system:\n")
        self.configs = {}
        for precision in precisions:
            print_maybe(f"{model_name}, {precision} precision")
            try:
                ask_for_patience("looking up best configuration")
                x = find_best_config(look_up_data, precision, memory / numa_nodes, num_threads, True)
                clean_line()
            except LookupError:
                clean_line()
                if no_interactive:
                    print(f"{model_name}, {precision} precision")
                print_red("Not enough resources on the system to run\n")
                continue
            self.configs[precision] = {"latency": x}
            num_proc = x["num_proc"] * numa_nodes
            print_maybe("Case minimizing latency:")
            print_maybe(f"{INDENT}best setting: {num_proc} x {x['num_threads']} x {x['bs']} [streams x threads x bs]")
            print_maybe(f"{INDENT}total throughput: {round(numa_nodes * x['total_throughput'], 2)} ips")
            latency_msg = f"{INDENT}latency: {round(1000. / x['throughput_per_unit'], 2)} ms"
            if num_proc > 1:
                latency_msg += f" [{num_proc} parallel streams each offering this latency]"
            print_maybe(latency_msg)
            print_maybe(f"{INDENT}memory usage: <{round(numa_nodes * x['memory'], 2)} GiB")
            ask_for_patience("looking up best configuration")
            x = find_best_config(look_up_data, precision, memory / numa_nodes, num_threads, False)
            clean_line()
            self.configs[precision]["throughput"] = x
            num_proc = x['num_proc'] * numa_nodes
            print_maybe("Case maximizing throughput:")
            print_maybe(f"{INDENT}best setting: {num_proc} x {x['num_threads']} x {x['bs']} [streams x threads x bs]")
            print_maybe(f"{INDENT}total throughput: {round(numa_nodes * x['total_throughput'], 2)} ips")
            print_maybe(f"{INDENT}memory usage: <{numa_nodes * round(x['memory'], 2)} GiB\n")

    def get_results(self):
        for values in self._results.values():
            if len(values) > 0:
                return self._results
        else:
            return None

    def _validate(self, numa_nodes, num_threads):
        raise NotImplementedError

    def _run_benchmark(self, get_cmd, start_delay=0):
        warm_up_completed = False
        for precision in self.precisions:
            try:
                configs = self.configs[precision]
            except KeyError:
                continue

            if precision == "fp16":
                os.environ["AIO_IMPLICIT_FP16_TRANSFORM_FILTER"] = ".*"

            if (self.numa_nodes > 1 or configs["latency"]["num_proc"] > 1) and not warm_up_completed:
                if run_benchmark(
                        get_cmd("warm_up"), 1, self.num_threads, 1, self.num_threads
                ) is None:
                    continue
                warm_up_completed = True

            print(f"{self.model_name}, {precision} precision")
            print("Case minimizing latency:")
            num_proc = self.numa_nodes * configs["latency"]["num_proc"]
            print(f"{INDENT}setting: {num_proc} x {configs['latency']['num_threads']} x {configs['latency']['bs']} "
                  f"[streams x threads x bs]")

            result = run_benchmark(
                get_cmd("latency", configs['latency']), self.numa_nodes, self.num_threads,
                configs["latency"]["num_proc"], configs["latency"]["num_threads"], start_delay=start_delay)
            if result is not None:
                throughput = round(result, 2)
                print(f"{INDENT}total throughput: {throughput} ips")
                throughput_per_unit = (
                        result / (configs['latency']['bs'] * self.numa_nodes * configs['latency']['num_proc']))  # noqa
                latency = round(1000. / throughput_per_unit, 2)
                latency_msg = f"{INDENT}latency: {latency} ms"
                if num_proc > 1:
                    latency_msg += f" [{num_proc} parallel streams each offering this latency]"
                print(latency_msg)
                self._results[precision]["latency"] = {
                    "config": {
                        "streams": num_proc,
                        "threads": configs['latency']['num_threads'],
                        "batch_size": configs['latency']['bs']},
                    "throughput": throughput,
                    "latency_ms": latency
                }

            print("Case maximizing throughput:")
            num_proc = self.numa_nodes * configs["throughput"]['num_proc']
            print(f"{INDENT}setting: {num_proc} x {configs['throughput']['num_threads']} x "
                  f"{configs['throughput']['bs']} [streams x threads x bs]")
            result = run_benchmark(
                get_cmd("latency", configs['throughput']), self.numa_nodes, self.num_threads,
                configs["throughput"]['num_proc'], configs["throughput"]['num_threads'], start_delay=start_delay)
            if result is not None:
                throughput = round(result, 2)
                print(f"{INDENT}total throughput: {throughput} ips\n")
                self._results[precision]["throughput"] = {
                    "config": {
                        "streams": num_proc,
                        "threads": configs['throughput']['num_threads'],
                        "batch_size": configs['throughput']['bs']},
                    "throughput": throughput
                }

            if precision == "fp16":
                os.environ["AIO_IMPLICIT_FP16_TRANSFORM_FILTER"] = ""

        press_enter_to_continue()


class YOLO(Runner):
    model_name = "YOLO v8s"
    precisions = ["fp32", "fp16"]

    def __init__(self, system, numa_nodes, num_threads, memory):
        super().__init__(system, self.model_name, numa_nodes, num_threads, memory, self.precisions)
        if len(self.configs) > 0 and get_bool_answer("Do you want to run actual benchmark to validate?"):
            self._validate(numa_nodes, num_threads)

    def _download_maybe(self):
        from utils.downloads.utils import get_downloads_path
        dataset_dir = os.path.join(get_downloads_path(), "aio_objdet_dataset")
        if not os.path.exists(dataset_dir):
            subprocess.run(["wget", "-P", "/tmp",
                            "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/aio_objdet_dataset.tar.gz"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["tar", "-xf", "/tmp/aio_objdet_dataset.tar.gz", "-C", get_downloads_path()],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(["rm", "/tmp/aio_objdet_dataset.tar.gz"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        os.environ["COCO_IMG_PATH"] = dataset_dir
        os.environ["COCO_ANNO_PATH"] = os.path.join(dataset_dir, "annotations.json")
        target_dir = os.path.join(get_downloads_path(), "yolov8s.pt")
        if not os.path.exists(target_dir):
            ask_for_patience(f"downloading {self.model_name} model")
            subprocess.run(["wget", "-P", get_downloads_path(),
                            "https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            clean_line()
        return target_dir

    def _validate(self, numa_nodes, num_threads):
        model_filepath = self._download_maybe()
        path_to_runner = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      "computer_vision/object_detection/yolo_v8/run.py")

        def get_cmd(scenario, config=None):
            if scenario == "warm_up":
                return f"{path_to_runner} -m {model_filepath} -p fp32 -f pytorch -b 1 --timeout={DAY_IN_SEC}"
            elif scenario in ["latency", "throughput"]:
                return (f"{path_to_runner} -m {model_filepath} -p fp32 -f pytorch -b {config['bs']} "
                        f"--timeout={DAY_IN_SEC}")
            else:
                assert False

        self._run_benchmark(get_cmd, start_delay=5)


class ResNet50(Runner):
    model_name = "ResNet-50 v1.5"
    precisions = ["fp32", "fp16"]

    def __init__(self, system, numa_nodes, num_threads, memory):
        super().__init__(
            system, self.model_name, numa_nodes, num_threads, memory, self.precisions)
        if len(self.configs) > 0 and get_bool_answer("Do you want to run actual benchmark to validate?"):
            self._validate(numa_nodes, num_threads)

    def _validate(self, numa_nodes, num_threads):
        path_to_runner = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      "computer_vision/classification/resnet_50_v15/run.py")

        def get_cmd(scenario, config=None):
            if scenario == "warm_up":
                return f"{path_to_runner} -m resnet50 -p fp32 -f pytorch -b 1 --timeout={DAY_IN_SEC}"
            elif scenario in ["latency", "throughput"]:
                return f"{path_to_runner} -m resnet50 -p fp32 -f pytorch -b {config['bs']} --timeout={DAY_IN_SEC}"
            else:
                assert False

        self._run_benchmark(get_cmd)


class BERT(Runner):
    model_name = "BERT large"
    precisions = ["fp32", "fp16"]

    def __init__(self, system, numa_nodes, num_threads, memory):
        super().__init__(
            system, self.model_name, numa_nodes, num_threads, memory, self.precisions)
        if len(self.configs) > 0 and get_bool_answer("Do you want to run actual benchmark to validate?"):
            self._validate(numa_nodes, num_threads)

    def _download_maybe(self):
        from utils.downloads.utils import get_downloads_path
        filename = "bert_large_mlperf.pt"
        target_dir = os.path.join(get_downloads_path(), filename)
        if not os.path.exists(target_dir):
            ask_for_patience(f"downloading {self.model_name} model")
            subprocess.run(["wget", "-O", target_dir,
                            "https://zenodo.org/records/3733896/files/model.pytorch?download=1"],
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            clean_line()
        return target_dir

    def _validate(self, numa_nodes, num_threads):
        model_filepath = self._download_maybe()
        path_to_runner = os.path.join(
            os.path.dirname(os.path.realpath(__file__)),
            "natural_language_processing/extractive_question_answering/bert_large/run_mlperf.py")

        def get_cmd(scenario, config=None):
            if scenario == "warm_up":
                return f"{path_to_runner} -m {model_filepath} -p fp32 -f pytorch -b 1 --timeout={DAY_IN_SEC}"
            elif scenario in ["latency", "throughput"]:
                return (f"{path_to_runner} -m {model_filepath} -p fp32 -f pytorch -b {config['bs']} "
                        f"--timeout={DAY_IN_SEC}")
            else:
                assert False

        self._run_benchmark(get_cmd)


class DLRM(Runner):
    model_name = "DLRM"
    precisions = ["fp32", "fp16"]

    def __init__(self, system, numa_nodes, num_threads, memory):
        super().__init__(
            system, self.model_name, numa_nodes, num_threads, memory, self.precisions)
        if len(self.configs) > 0 and get_bool_answer("Do you want to run actual benchmark to validate?"):
            self._validate(numa_nodes, num_threads)

    def _validate(self, numa_nodes, num_threads):
        path_to_runner = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "recommendation/dlrm_torchbench/run.py")

        def get_cmd(scenario, config=None):
            if scenario == "warm_up":
                return f"{path_to_runner} -p fp32 -f pytorch -b 1024 --timeout={DAY_IN_SEC}"
            elif scenario in ["latency", "throughput"]:
                return f"{path_to_runner} -p fp32 -f pytorch -b {config['bs']} --timeout={DAY_IN_SEC}"
            else:
                assert False

        self._run_benchmark(get_cmd)


class Whisper(Runner):
    model_name = "Whisper medium EN"
    precisions = ["fp32", "fp16"]

    def __init__(self, system, numa_nodes, num_threads, memory):
        super().__init__(
            system, self.model_name, numa_nodes, num_threads, memory, self.precisions)
        if len(self.configs) > 0 and get_bool_answer("Do you want to run actual benchmark to validate?"):
            self._validate(numa_nodes, num_threads)

    def _validate(self, numa_nodes, num_threads):
        path_to_runner = os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "speech_recognition/whisper/run.py")

        def get_cmd(scenario, config=None):
            return f"{path_to_runner} -m medium.en --timeout={DAY_IN_SEC}"

        self._run_benchmark(get_cmd)


def convert_name(text):
    return text.replace(" ", "_").replace("-", "_").lower()


def main():
    models = [ResNet50, YOLO, BERT, DLRM, Whisper]
    parser = argparse.ArgumentParser(prog="AML benchmarking tool")
    parser.add_argument("--no-interactive", action="store_true", help="don't ask for user input")
    parser.add_argument("--model", type=str, choices=[convert_name(model.model_name) for model in models],
                        help="choose a single model to evaluate")
    parser.add_argument("--system", type=str, choices=[convert_name(system) for system in SYSTEMS.keys()],
                        help="specify Ampere CPU")
    parser.add_argument("--memory", type=int, help="limit memory to a specified value [GiB]")
    parser.add_argument("--max-threads", type=int, help="limit number of threads to use per NUMA node")
    args = parser.parse_args()
    global no_interactive
    no_interactive = args.no_interactive

    is_setup_done()
    system, numa_nodes, num_threads, memory = identify_system(args)
    results_all = {}
    for model in models:
        if args.model is not None and convert_name(model.model_name) != args.model:
            continue
        results = model(system, numa_nodes, num_threads, memory).get_results()
        if results is not None:
            results_all[model.model_name] = results
    if len(results_all) > 0:
        filename = "evaluation_results.json"
        print(f"Dumping results to {filename} file.")
        with open(filename, "w") as f:
            json.dump(results_all, f)
    print_green("Evaluation finished.")


if __name__ == "__main__":
    main()
