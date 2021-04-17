import os
import sys


def get_env_variable(env_var_name: str, fail_message: str):
    """
    A function checking the value of environment variable and returning given fail message if not set.

    :param env_var_name: str
    :param fail_message: str
    """
    try:
        return os.environ[env_var_name]
    except KeyError:
        print_goodbye_message_and_die(fail_message)


def print_goodbye_message_and_die(message):
    """
    A function printing fail message and making program quit with exit code 1.

    :param message: str
    """
    print(f"\nFAIL: {message}")
    sys.exit(1)


def print_warning_message(message):
    """
    A function printing a warning message but not killing the program.

    :param message: str
    """
    print(f"\nCAUTION: {message}")
