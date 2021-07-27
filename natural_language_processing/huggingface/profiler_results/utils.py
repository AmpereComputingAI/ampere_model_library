import os


def get_profiler_results_path():
    """
    A function returning absolute path to profiler results dir.
    :return: str, path to downloads dir
    """
    return os.path.dirname(os.path.realpath(__file__))
