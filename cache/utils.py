import os


def get_cache_dir():
    return os.path.dirname(os.path.realpath(__file__))
