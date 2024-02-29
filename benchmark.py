import os
import sys

LATEST_VERSION = "2.1.0a0+gite0a1120"
SYSTEMS = {
    "Altra": {},
    "Altra Max": {},
    "AmpereOne": {},
    "AmpereOneX": {},
}


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
    if not os.path.exists(os.path.join(os.path.dirname(os.path.realpath(__file__)), ".setup_completed")):
        do_the_setup_message()
    try:
        from utils.perf_prediction.predictor import predict
    except ImportError:
        do_the_setup_message()
    print_green("Setup verified. You are good to go! 🔥")


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
    memory_total = mem.total / 1024**3
    memory_available = mem.available / 1024**3

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
        print(f"Sockets: {num_sockets}\nThreads: {num_threads_per_socket}\nMemory: {memory_total} [GiB]\n")

    answer = "no"
    affirmative = ["y", "Y", "yes", "YES"]
    negative = ["n", "N", "no", "NO"]
    if system is not None:
        system_identifed_as()
        answer = input("Is this correct? (y/n)").strip()
        while answer not in affirmative + negative:
            answer = input("Is this correct? (y/n)").strip()
    else:
        print_red("\nCouldn't identify system. Are you running this on Ampere CPU?")

    if answer in negative:
        print("\nPlease select your system from the following list:")
        system_map = {}
        for i, system in enumerate(SYSTEMS.keys()):
            system_map[i] = system
            print(f"{''*3}{i}: {system}")
        print()
        answer = None
        while answer not in system_map.keys():
            try:
                answer = int(input(f"Input number for your system [0-{len(system_map)-1}]"))
            except ValueError:
                pass
        system = system_map[answer]
        system_identifed_as()



def main():
    is_setup_done()
    identify_system()


if __name__ == "__main__":
    main()
