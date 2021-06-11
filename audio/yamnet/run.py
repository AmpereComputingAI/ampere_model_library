import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import time
import os
import argparse

from scipy.io import wavfile
from scipy.signal import resample
from utils.yamnet import Yamnet
from utils.benchmark import run_model
from utils.tf import TFSavedModelRunner


def parse_args():
    parser = argparse.ArgumentParser(description="Knowles benchmark.")
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

    def run_single_pass(yamnet_model, yamnet):

        scores, class_map_path = yamnet_model.run_from_hub(yamnet.get_input_array())
        yamnet.submit_predictions(scores, class_map_path)

    dataset = Yamnet(sound_path, labels_path)
    runner = TFSavedModelRunner(url_to_model='https://tfhub.dev/google/yamnet/1')

    return run_model(run_single_pass, runner, dataset, 1, num_of_runs, timeout)


if __name__ == "__main__":
    args = parse_args()
    if args.precision == "fp32":
        run_yamnet(args.num_runs, args.timeout, args.sound_path, args.labels_path)
