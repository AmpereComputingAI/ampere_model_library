# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
from pathlib import Path


def get_downloads_path():
    """
    A function returning absolute path to downloads dir.
    :return: str, path to downloads dir
    """
    cache_dir = Path(Path.home(), ".cache/ampere_model_library")
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir
    # return os.path.dirname(os.path.realpath(__file__))
