import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
import time
import os
import argparse

from IPython.display import Audio
from scipy.io import wavfile
from scipy.signal import resample


# rsync for graviton
# alias temp_sync=
# "rsync -a --exclude "__pycache__/" /home/marcel/dev/knowles_benchmark/ graviton:/home/marcel/dev/knowles_benchmark/"


def parse_args():
    parser = argparse.ArgumentParser(description="Knowles benchmark.")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32", "fp16", "int8"], required=True,
                        help="precision of the model provided")
    return parser.parse_args()


def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with tf.io.gfile.GFile(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])

    return class_names


def ensure_sample_rate(original_sample_rate, waveform,
                       desired_sample_rate=16000):
    """Resample waveform if required."""
    if original_sample_rate != desired_sample_rate:
        desired_length = int(round(float(len(waveform)) /
                                   original_sample_rate * desired_sample_rate))
        waveform = resample(waveform, desired_length)
    return desired_sample_rate, waveform


def run_tf_fp32():

    # model documentation: https://tfhub.dev/google/yamnet/1

    # Load the model.
    model1 = TFSavedModelRunner()
    model = hub.load('https://tfhub.dev/google/yamnet/1')

    class_map_path = model.class_map_path().numpy()
    class_names = class_names_from_csv(class_map_path)

    total_time = 0.0
    count = 0

    for file in os.listdir('/onspecta/dev/knowles_benchmark/sounds/'):
        if file.endswith('.wav'):
            wav_file_name = 'sounds/' + file
            print(wav_file_name)
            sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
            sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)

            # The wav_data needs to be normalized to values in [-1.0, 1.0]
            waveform = wav_data / tf.int16.max

            # Run the model, check the output.
            start = time.time()
            scores, embeddings, spectrogram = model(waveform)
            finish = time.time()

            inference_time = finish - start
            total_time += inference_time

            scores_np = scores.numpy()
            spectrogram_np = spectrogram.numpy()
            infered_class = class_names[scores_np.mean(axis=0).argmax()]
            print(f'The main sound is: {infered_class}')

            count += 1
            continue

    print("*" * 45)
    print(f'Total time was: {total_time} seconds')
    print(f'Average inference time was: {total_time/count} seconds')
    print(f'the measure was performed on: {count} samples')


if __name__ == "__main__":
    args = parse_args()
    if args.precision == "fp32":
        run_tf_fp32(
            args.model_path, args.num_runs, args.timeout, args.sounds_path, args.labels_path
        )

