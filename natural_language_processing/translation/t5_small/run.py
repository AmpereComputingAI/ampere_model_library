import argparse
import time
import os
import json
import csv
from utils.nlp.fastT5.fastT5 import get_onnx_model
from transformers import AutoTokenizer
import utils.benchmark as bench_utils
from utils.nlp.opus import Opus
from utils.misc import advertise_aio
import onnxruntime as ort


def parse_args():
    parser = argparse.ArgumentParser(description="Run T5 small architecture for translation.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=True,
                        help="path to the directory containing .onnx models")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    return parser.parse_args()


class Runner:
    def __init__(self, model):
        try:
            ort.AIO
        except AttributeError:
            advertise_aio("ONNXRunTime")

        self._model = model

        self._times_invoked = 0
        self._start_times = list()
        self._finish_times = list()

        print("\nRunning with ONNX Runtime\n")

    def run(self, opus):
        start = time.time()
        input_data = opus.get_input()
        output = self._model.generate(
            input_ids=input_data["input_ids"],
            attention_mask=input_data["input_ids"],
            num_beams=1,
            max_length=128
        )
        opus.submit_prediction(output)
        finish = time.time()

        self._start_times.append(start)
        self._finish_times.append(finish)
        self._times_invoked += 1

    def print_performance_metrics(self, batch_size):
        """
        A function printing performance metrics on runs executed by the runner so far.
        param batch_size: int, batch size - if batch size was varying over the runs an average should be supplied
        """
        perf = bench_utils.print_performance_metrics(
            self._start_times, self._finish_times, self._times_invoked, batch_size)
        if os.getenv("AIO_PROFILER", "0") == "1":
            ort.AIO.print_profile_data()
        if os.getenv("ORT_PROFILER", "0") == "1":
            profs = [self._model.encoder.encoder.end_profiling(),
                     self._model.decoder.decoder.end_profiling(),
                     self._model.decoder_init.decoder.end_profiling()]

            if not os.path.exists("profiler_output/ort/"):
                os.makedirs("profiler_output/ort/")
            for i in range(len(profs)):
                if profs[i] not in profs[:i]:
                    os.replace(profs[i], f"profiler_output/ort/{profs[i]}")

        dump_dir = os.environ.get("RESULTS_DIR")
        if dump_dir is not None and len(self._start_times) > 2:
            with open(f"{dump_dir}/meta_{os.getpid()}.json", "w") as f:
                json.dump({"batch_size": batch_size}, f)
            with open(f"{dump_dir}/{os.getpid()}.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(self._start_times[2:])
                writer.writerow(self._finish_times[2:])

        return perf


def run_ort(model_path, batch_size, num_runs, timeout, **kwargs):
    def run_single_pass(external_runner, opus):
        external_runner.run(opus)

    model_name = "t5-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(text):
        return tokenizer(text, return_tensors='pt', padding='max_length', max_length=16, truncation=True)

    def detokenize(output_tokens):
        return tokenizer.batch_decode(output_tokens, skip_special_tokens=True)

    dataset = Opus(batch_size, tokenize, detokenize)

    return bench_utils.run_model(run_single_pass, Runner(get_onnx_model(model_name, model_path, quantized=False)),
                                 dataset, batch_size, num_runs, timeout)


def run_ort_fp32(model_path, batch_size, num_runs, timeout, **kwargs):
    return run_ort(model_path, batch_size, num_runs, timeout, **kwargs)


if __name__ == "__main__":
    run_ort(**vars(parse_args()))
