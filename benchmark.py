import os
import sys
import json
import subprocess
import urllib.request

LATEST_VERSION = "2.1.0a0+gite0a1120"
SYSTEMS = {
    "Altra": {
        "ResNet-50 v1.5": "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/q80_30%40ampere_pytorch_1.10.0%40resnet_50_v1.5.json"
    },
    "Altra Max": {},
    "AmpereOne": {},
    "AmpereOneX": {},
}

AFFIRMATIVE = ["y", "Y", "yes", "YES"]
NEGATIVE = ["n", "N", "no", "NO"]

MAX_DEVIATION = 0.01  # max deviation [abs((value_n+1 / value_n) - 1.)] between sample n and sample n+1
MIN_MEASUREMENTS_IN_OVERLAP_COUNT = 10
DAY_IN_SEC = 60*60*24

debug = os.environ.get("DEBUG") == "1"


def print_red(message):
    print(f"\033[91m{message}\033[0m")


def print_green(message):
    print(f"\033[92m{message}\033[0m")


def go_ampere_message():
    print_red(f"\nThis script requires latest Ampere optimized PyTorch ({LATEST_VERSION}) to be installed. "
              "\nConsider using our Docker images available at https://hub.docker.com/u/amperecomputingai")
    sys.exit(1)


def do_the_setup_message():
    print_red(f"\nBefore running this script please run `bash setup_deb.sh && source set_env_variables.sh`")
    sys.exit(1)


def is_setup_done():
    try:
        import torch
    except ImportError:
        go_ampere_message()
    if '_aio_profiler_print' not in dir(torch._C) or torch.__version__ != LATEST_VERSION:
        go_ampere_message()
    if torch.__version__ != "2.1.0a0+gite0a1120":
        go_ampere_message()
    setup_confirmation = os.path.join(os.path.dirname(os.path.realpath(__file__)), ".setup_completed")
    if not os.path.exists(setup_confirmation):
        do_the_setup_message()
    elif open(setup_confirmation, "r").read() != open("/etc/machine-id", "r").read():
        do_the_setup_message()
    if os.environ.get("PYTHONPATH") != os.path.dirname(os.path.realpath(__file__)):
        do_the_setup_message()
    print_green("Setup verified. You are good to go! 🔥")


def get_bool_answer(question):
    answer = None
    while answer not in AFFIRMATIVE + NEGATIVE:
        answer = input(f"{question} (y/n)").strip()
    return answer in AFFIRMATIVE


def identify_system():
    import psutil
    import subprocess
    from cpuinfo import get_cpu_info
    altra_flags = ['aes', 'asimd', 'asimddp', 'asimdhp', 'asimdrdm', 'atomics', 'cpuid', 'crc32', 'dcpop', 'evtstrm',
                   'fp', 'fphp', 'lrcpc', 'pmull', 'sha1', 'sha2', 'ssbs']
    aone_flags = ['aes', 'asimd', 'asimddp', 'asimdfhm', 'asimdhp', 'asimdrdm', 'atomics', 'bf16', 'bti', 'cpuid',
                  'crc32', 'dcpodp', 'dcpop', 'dit', 'ecv', 'evtstrm', 'fcma', 'flagm', 'flagm2', 'fp', 'fphp', 'frint',
                  'i8mm', 'ilrcpc', 'jscvt', 'lrcpc', 'paca', 'pacg', 'pmull', 'rng', 'sb', 'sha1', 'sha2', 'sha3',
                  'sha512', 'sm3', 'sm4', 'ssbs', 'uscat']

    cpu_info = get_cpu_info()
    flags = cpu_info["flags"]
    num_threads = cpu_info["count"]
    num_sockets = int([n for n in subprocess.check_output(["lscpu"]).decode().split("\n")
                       if "Socket(s):" in n][0].split()[1])  # so ugly
    num_threads_per_socket = num_threads // num_sockets
    mem = psutil.virtual_memory()
    memory_total = mem.total / 1024 ** 3
    memory_available = mem.available / 1024 ** 3

    if all([item in altra_flags for item in flags]):
        if num_threads_per_socket > 80:
            system = "Altra Max"
        else:
            system = "Altra"
    elif all([item in aone_flags for item in flags]):
        if num_threads_per_socket > 160:
            system = "AmpereOneX"
        else:
            system = "AmpereOne"
    else:
        system = None

    def system_identifed_as():
        print(f"\nSystem identified as {system}")
        print(f"Sockets: {num_sockets}\nThreads: {num_threads_per_socket}\nMemory: {round(memory_total, 2)} [GiB]\n")

    run_selection = True
    if system is not None:
        system_identifed_as()
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
        system_identifed_as()

    return system, num_sockets, num_threads_per_socket, memory_available


def gen_threads_config(num_threads, process_id):
    threads_to_use = [str(t) for t in online_threads[num_threads * process_id:num_threads * (process_id + 1)]]
    assert len(threads_to_use) == num_threads
    return " ".join(threads_to_use), ",".join(threads_to_use)


class Results:
    def __init__(self, results_dir, processes_count):
        self._results_dir = results_dir
        self._prev_measurements_count = None
        self._prev_throughput_total = None
        self._processes_count = processes_count
        self.stable = False

    def calculate_throughput(self, final_calc=False):
        from filelock import FileLock
        logs = [log for log in os.listdir(self._results_dir) if "json" in log and "lock" not in log]
        assert len(logs) == self._processes_count

        loaded_logs = []
        for log in logs:
            log_filepath = os.path.join(self._results_dir, log)
            with FileLock(f"{log_filepath}.lock", timeout=60):
                with open(log_filepath, "r") as f:
                    loaded_logs.append(json.load(f))

        measurements_counts = [(len(log["start_times"]), len(log["finish_times"]), len(log["workload_size"])) for log in
                               loaded_logs]
        assert all(x[0] == x[1] == x[2] and x[0] >= MIN_MEASUREMENTS_IN_OVERLAP_COUNT for x in measurements_counts)
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
            assert measurements_completed_in_overlap >= MIN_MEASUREMENTS_IN_OVERLAP_COUNT
            measurements_completed_in_overlap_total += measurements_completed_in_overlap
            throughput_total += input_size_processed_per_process / total_latency_per_process

        if self._prev_measurements_count is not None and \
                measurements_completed_in_overlap_total > self._prev_measurements_count:
            self.stable = abs((throughput_total / self._prev_throughput_total) - 1.) <= MAX_DEVIATION
        self._prev_throughput_total = throughput_total
        self._prev_measurements_count = measurements_completed_in_overlap_total

        if not self.stable and not final_calc:
            print("Result not yet stable - current throughput: {:.2f} fps".format(throughput_total))

        return throughput_total


def run_benchmark(model_script, num_sockets, num_proc, num_threads):
    os.environ["IGNORE_DATASET_LIMITS"] = "1"

    os.environ["AIO_NUM_THREADS"] = str(num_threads)
    os.environ["OMP_NUM_THREADS"] = str(num_threads)

    results_dir = os.path.join(os.getcwd(), ".cache_aml")
    os.environ["RESULTS_DIR"] = results_dir
    if os.path.exists(results_dir) and os.path.isdir(results_dir):
        for filepath in os.listdir(results_dir):
            os.remove(os.path.join(results_dir, filepath))
    else:
        os.mkdir(results_dir)

    results = Results(results_dir, num_proc)
    current_subprocesses = list()
    for n in range(num_proc):
        aio_numa_cpus, physcpubind = gen_threads_config(num_threads, n)
        os.environ["AIO_NUMA_CPUS"] = aio_numa_cpus
        os.environ["DLS_NUMA_CPUS"] = aio_numa_cpus
        cmd = ["numactl", f"--physcpubind={physcpubind}",
               "python3"] + model_script.split()
        if debug:
            current_subprocesses.append(subprocess.Popen(cmd))
        else:
            current_subprocesses.append(subprocess.Popen(
                cmd, stdout=open(os.devnull, 'wb'), stderr=open(os.devnull, 'wb')))

    failure = False
    while not all(p.poll() is not None for p in current_subprocesses):
        time.sleep(15)
        try:
            results.calculate_throughput()
        except AssertionError:
            pass
        failure = any(p.poll() is not None and p.poll() != 0 for p in current_subprocesses)
        if results.stable or failure:
            Path(os.path.join(results_dir, "STOP")).touch()
            break

    if not failure:
        # wait for subprocesses to finish their job if all are alive till now
        failure = any(p.wait() != 0 for p in current_subprocesses)

    if failure:
        print_red("FAIL: At least one process returned exit code other than 0 or died!")
        sys.exit(1)

    values = results.calculate_throughput(final_calc=True)


class Runner:
    def __init__(self, system, model_name, num_sockets, num_threads, memory, precisions):
        with urllib.request.urlopen(SYSTEMS[system][model_name]) as url:
            look_up_data = json.load(url)
        from utils.perf_prediction.predictor import find_best_config
        self.configs = {}
        for precision in precisions:
            print(f"{model_name}, {precision} precision")
            x = find_best_config(look_up_data, precision, memory, num_threads, True)
            if x[4] is None:
                print_red("Not enough memory on the system to run\n")
                continue
            self.configs[precision] = {"latency": x}
            num_proc = x[1] * num_sockets
            print("Case minimizing latency:")
            print(f"{' ' * 3}best setting: {x[0]} x {num_proc} x {x[2]} [bs x num_proc x num_threads]")
            print(f"{' ' * 3}total throughput: {round(num_sockets * x[3], 2)} ips")
            print(f"{' ' * 3}latency: {round(1000. / x[4], 2)} ms")
            print(f"{' ' * 3}memory usage: <{round(num_sockets * x[5], 2)} GiB")
            x = find_best_config(look_up_data, precision, memory, num_threads, False)
            self.configs[precision]["throughput"] = x
            num_proc = x[1] * num_sockets
            print("Case maximizing throughput:")
            print(f"{' ' * 3}best setting: {x[0]} x {num_proc} x {x[2]} [bs x num_proc x num_threads]")
            print(f"{' ' * 3}total throughput: {round(num_sockets * x[3], 2)} ips")
            print(f"{' ' * 3}memory usage: <{num_sockets * round(x[5], 2)} GiB\n")


class ResNet50(Runner):
    def __init__(self, system, num_sockets, num_threads, memory):
        self.model_name = "ResNet-50 v1.5"
        self.precisions = ["fp32", "fp16"]
        super().__init__(
            system, self.model_name, num_sockets, num_threads, memory, self.precisions)
        if len(self.configs) > 0 and get_bool_answer("Do you want to run actual benchmark to validate?"):
            self._validate(num_sockets)

    def _validate(self, num_sockets):
        path_to_runner = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                      "computer_vision/classification/resnet_50_v15/run.py")
        for precision in self.precisions:
            print(f"{self.model_name}, {precision} precision")
            try:
                configs = self.configs[precision]
            except KeyError:
                continue
            print("Case minimizing latency:")
            cmd = f"{path_to_runner} -m resnet50 -p fp32 -f pytorch -b {configs['latency'][0]} --timeout={DAY_IN_SEC}"
            run_benchmark(cmd, num_sockets, configs["latency"][1], configs["latency"][2])


def main():
    is_setup_done()
    system, num_sockets, num_threads, memory = identify_system()
    print("\nExpected performance on your system:\n")
    for model in [ResNet50]:
        model(system, num_sockets, num_threads, memory)


if __name__ == "__main__":
    main()
