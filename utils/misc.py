# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import os
import sys
import hashlib


class UnsupportedPrecisionValueError(ValueError):
    """
    An error being raised when requested precision is not available.
    """
    pass


class ModelPathUnspecified(Exception):
    """
    An Exception class being raised when model name is unspecified.
    """
    pass


class OutOfInstances(Exception):
    """
    An exception class being raised as an error in case of lack of further images to process by the pipeline.
    """
    pass


class FrameworkUnsupportedError(Exception):
    """
    An exception class being raised as an error in case of lack of implemented framework pipeline.
    """
    pass


def get_hash_of_a_file(path_to_file):
    """
    A function calculating md5 hash for a file under the supplied path.
    :param path_to_file: str, path to a file of which a hash should be calculated
    :return: string containing the md5 hash of a file
    """
    hash_md5 = hashlib.md5()
    with open(path_to_file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


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
    print(f"\n\033[91mFAIL: {message}\033[0m")
    sys.exit(1)


def print_warning_message(message):
    """
    A function printing a warning message but not killing the program.
    :param message: str
    """
    print(f"\n\033[91mCAUTION: {message}\033[0m")


def advertise_aio(framework_name):
    print(f"\n\033[91mYou are running {framework_name} missing Ampere optimizations.\nConsider using AI-dedicated Docker images for increased performance.\nAvailable at: https://solutions.amperecomputing.com/solutions/ampere-ai\n\033[0m")
