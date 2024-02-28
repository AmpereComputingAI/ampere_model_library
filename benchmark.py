import os
import sys

LATEST_VERSION = "2.1.0a0+gite0a1120"


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
    pass


def main():
    is_setup_done()
    identify_system()


if __name__ == "__main__":
    main()
