# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import argparse
import os
from typing import List
from utils.misc import print_warning_message

SUPPORTED_FRAMEWORKS = ["tf", "ort", "pytorch", "ctranslate2", "tflite"]


class DefaultArgParser:
    def __init__(self, supported_frameworks: List[str], default_timeout=60.):
        self.parser = argparse.ArgumentParser(prog="AML model-dedicated runner")

        if len(supported_frameworks) >= 2:
            for framework in supported_frameworks:
                assert framework in SUPPORTED_FRAMEWORKS, \
                    f"{framework} is not listed as globally supported [{SUPPORTED_FRAMEWORKS}]"
            self.parser.add_argument("-f", "--framework", type=str, choices=supported_frameworks, required=True)
        self.parser.add_argument("--timeout", type=float, default=default_timeout,
                                 help="timeout in seconds")
        self.parser.add_argument("--num_runs", type=int,
                                 help="number of inference calls to execute")

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def ask_for_batch_size(self, default_batch_size=1):
        self.parser.add_argument("-b", "--batch_size", type=int, default=default_batch_size,
                                 help="batch size to feed the model with")

    def require_model_path(self):
        self.parser.add_argument("-m", "--model_path", type=str, required=True)

    def require_model_name(self, choices: List[str]):
        self.parser.add_argument("-m", "--model_name", type=str, choices=choices, required=True)

    def parse(self):
        return self.parser.parse_args()


class Dataset:
    available_instances = None

    def do_skip(self) -> bool:
        return os.environ.get("IGNORE_DATASET_LIMITS") == "1"

    def reset(self) -> bool:
        raise NotImplementedError

    def _summarize_accuracy(self) -> dict:
        raise NotImplementedError

    # don't override this method, override _summarize_accuracy instead
    def summarize_accuracy(self) -> dict:
        accuracy_results = self._summarize_accuracy()
        if accuracy_results is None:
            accuracy_results = {}
        assert type(accuracy_results) is dict
        for k, v in accuracy_results.items():
            assert isinstance(k, str)
            try:
                float(v)
            except Exception as e:
                raise e
        return accuracy_results

    def print_accuracy_metrics(self) -> dict:
        accuracy_results = self.summarize_accuracy()
        if len(accuracy_results) == 0:
            print_warning_message("No accuracy metrics to print.")
        else:
            max_len = 14
            indent = 2 * " "
            print(f"\n{indent}ACCURACY")
            for metric in accuracy_results.keys():
                print(f"{3 * indent}{metric[:max_len]}{(max_len - len(metric)) * ' '}{3 * indent}" +
                      "= {:>7.3f}".format(float(accuracy_results[metric])))
        return accuracy_results


class DummyDataset(Dataset):
    def __init__(self):
        import sys
        self.available_instances = int(sys.maxsize)

    def reset(self) -> bool:
        return True

    def _summarize_accuracy(self) -> dict:
        return {}
