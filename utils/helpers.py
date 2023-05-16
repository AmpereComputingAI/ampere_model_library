import argparse
from typing import List


SUPPORTED_FRAMEWORKS = ["tf", "ort", "pytorch", "ctranslate2", "tflite"]


class DefaultArgParser:
    def __init__(self, supported_frameworks: List[str]):
        self.parser = argparse.ArgumentParser(prog=f"AML model-dedicated runner")

        if len(supported_frameworks) >= 2:
            for framework in supported_frameworks:
                assert framework in SUPPORTED_FRAMEWORKS, \
                    f"{framework} is not listed as globally supported [{SUPPORTED_FRAMEWORKS}]"
            self.parser.add_argument("-f", "--framework", type=str, choices=supported_frameworks, required=True)
        self.parser.add_argument("-b", "--batch_size", type=int, default=1,
                                 help="batch size to feed the model with")
        self.parser.add_argument("--timeout", type=float, default=60.0,
                                 help="timeout in seconds")
        self.parser.add_argument("--num_runs", type=int,
                                 help="number of inference calls to execute")

    def add_argument(self, *args, **kwargs):
        self.parser.add_argument(*args, **kwargs)

    def require_model_path(self):
        self.parser.add_argument("-m", "--model_path", type=str, required=True)

    def parse(self):
        return self.parser.parse_args()


class DatasetStub:
    def reset(self):
        raise NotImplemented

    def get_inputs(self):
        raise NotImplemented

    def submit_predictions(self, prediction):
        raise NotImplemented

    def summarize_accuracy(self):
        raise NotImplemented
