# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import os


def get_downloads_path():
    """
    A function returning absolute path to downloads dir.
    :return: str, path to downloads dir
    """
    return os.path.dirname(os.path.realpath(__file__))
