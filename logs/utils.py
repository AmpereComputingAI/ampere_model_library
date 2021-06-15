import os


def get_logs_path():
    """
    A function returning absolute path to logs dir.
    :return: str, path to downloads dir
    """
    return os.path.dirname(os.path.realpath(__file__))
