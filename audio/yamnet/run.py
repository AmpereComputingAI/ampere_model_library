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


# rsync for graviton
# alias temp_sync=
# "rsync -a --exclude "__pycache__/" /home/marcel/dev/knowles_benchmark/ graviton:/home/marcel/dev/knowles_benchmark/"


def parse_args():
    parser = argparse.ArgumentParser(description="Knowles benchmark.")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32", "int8", "test"], required=True,
                        help="precision of the model provided")
    parser.add_argument("--sound_path", type=str, required=False, help="For example: "
                                                                       "'sounds/test_sounds/accordion.wav'")
    parser.add_argument("--model_path", type=str, required=False, help="model path for tflite")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    return parser.parse_args()


def run_tf_fp32(sound_path):

    # Load the model.
    model = hub.load('https://tfhub.dev/google/yamnet/1')

    class_map_path = model.class_map_path().numpy()
    class_names = class_names_from_csv(class_map_path)

    wav_file_name = sound_path
    sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
    sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

    # The wav_data needs to be normalized to values in [-1.0, 1.0]
    waveform = wav_data / tf.int16.max

    if waveform.shape[0] > 16029:
        waveform_processed = waveform[:16029]
    elif waveform.shape[0] <= 130000:

        difference = 130000 - waveform.shape[0]
        empty_array = np.zeros(difference)
        waveform_processed = np.append(waveform, empty_array * 0, axis=0)

    scores, embeddings, spectrogram = model(waveform_processed)

    inference_time = finish - start
    total_time += inference_time
    count += 1

    scores_np = scores.numpy()
    infered_class = class_names[scores_np.mean(axis=0).argmax()]
    print(f'The main sound is: {infered_class}')


def run_yamnet(batch_size, num_of_runs, timeout, sounds_path):
    def run_single_pass(yamnet_model, yamnet):
        scores, embeddings, spectrogram = yamnet_model.run(yamnet.get_input_array())

    dataset = Yamnet(batch_size, sounds_path,
                     pre_processing="Yamnet")
    runner = hub.load('https://tfhub.dev/google/yamnet/1')

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


if __name__ == "__main__":
    args = parse_args()
    if args.precision == "fp32":
        run_tf_fp32(args.sound_path)
    elif args.precision == "int8":
        run_tf_int8(args.model_path)
    elif args.precision == "test":
        run_yamnet(args.batch_size, args.num_runs, args.timeout, args.sound_path)
