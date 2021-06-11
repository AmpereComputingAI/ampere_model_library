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
    parser.add_argument("--labels_path", type=str, required=False, help="path to labels")
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


def run_tf_fp32(sound_path):

    # Load the model.
    model = hub.load('https://tfhub.dev/google/yamnet/1')

    # class_map_path = model.class_map_path().numpy()
    # class_names = class_names_from_csv(class_map_path)

    yamnet_classes = yamnet_model.class_names('yamnet_class_map.csv')

    # input_array = np.empty([2, 130000])  # NHWC order
    # input_array[0] = sound1
    # input_array[1] = sound2

    # sound1 = '/onspecta/model_zoo/sounds/sounds/accordion.wav'
    # sound2 = 'sound/onspecta/model_zoo/sounds/sounds/bark.wav'

    # if self.__pre_processing:
    #     input_array = pp.pre_process(input_array, self.__pre_processing, self.__color_model)
    # return input_array
    # input_array = np.empty([2, 16029])  # NHWC order


    # for i in range(1, 3):
    # wav_file_name = sound_path + '0' + str(i) + '.wav'
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

        # input_array[i] = waveform_processed

    # for i in range(1, 3):
    scores, embeddings, spectrogram = model(waveform_processed)

    prediction = np.mean(scores, axis=0)
    top5_i = np.argsort(prediction)[::-1][:5]
    print('test', ':\n' +
          '\n'.join('  {:12s}: {:.3f}'.format(yamnet_classes[i], prediction[i])
                    for i in top5_i))

    # scores_np = scores.numpy()
    # infered_class = class_names[scores_np.mean(axis=0).argmax()]
    # print(f'The main sound is: {infered_class}')
        # continue


def run_yamnet(batch_size, num_of_runs, timeout, sound_path, labels_path):
    def run_single_pass(yamnet_model, yamnet):

        scores, class_map_path = yamnet_model.run_from_hub(yamnet.get_input_array())
        yamnet.submit_predictions(scores, class_map_path)

    dataset = Yamnet(batch_size, sound_path, labels_path)

    runner = TFSavedModelRunner(url_to_model='https://tfhub.dev/google/yamnet/1')

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


if __name__ == "__main__":
    args = parse_args()
    if args.precision == "benchmark":
        run_tf_fp32(args.sound_path)
    elif args.precision == "fp32":
        run_yamnet(args.batch_size, args.num_runs, args.timeout, args.sound_path, args.labels_path)
