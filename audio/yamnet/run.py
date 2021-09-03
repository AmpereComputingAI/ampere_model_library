import os
import argparse

from utils.audioset import AudioSet
from utils.benchmark import run_model
from audio.yamnet.runner import YamnetRunner
from utils.profiler_wrapper import TBTracer, print_prof

import tensorflow_hub as hub


def parse_args():
    parser = argparse.ArgumentParser(description="Yamnet benchmark.")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument("--sound_path", type=str, required=False, help="path to directory with audio files, e.g."
                                                                       "path/to/sounds/")
    parser.add_argument("--labels_path", type=str, required=False, help="path to labels, e.g. path/to/labels.csv")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    return parser.parse_args()


def run_yamnet(num_of_runs, timeout, sound_path, labels_path):
    #runner; dataset
    def run_single_pass(yamnet_runner, audioset):

        scores = yamnet_runner.run(audioset.get_input_array())
        audioset.submit_predictions(scores, yamnet_runner.get_class_map_path())

    dataset = AudioSet(sound_path, labels_path)
    runner = YamnetRunner(url_to_model='https://tfhub.dev/google/yamnet/1')

    return run_model(run_single_pass, runner, dataset, 1, num_of_runs, timeout)


if __name__ == "__main__":

    tracer = TBTracer()
    args = parse_args()
    if args.precision == "fp32":
        run_yamnet(args.num_runs, args.timeout, args.sound_path, args.labels_path)

    tracer.write()
