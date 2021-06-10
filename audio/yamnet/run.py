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
                        type=str, choices=["fp32", "int8"], required=True,
                        help="precision of the model provided")
    parser.add_argument("--sound_path", type=str, required=False, help="For example: "
                                                                       "'sounds/test_sounds/accordion.wav'")
    parser.add_argument("--model_path", type=str, required=False, help="model path for tflite")
    parser.add_argument("--intra",
                        type=int, default=1, required=False)
    parser.add_argument("--inter",
                        type=int, default=1, required=False)
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


def run_tf_fp32(sound_path, intra, inter):

    tf.config.threading.set_intra_op_parallelism_threads(intra)
    tf.config.threading.set_inter_op_parallelism_threads(inter)

    # Load the model.
    model = hub.load('https://tfhub.dev/google/yamnet/1')

    class_map_path = model.class_map_path().numpy()
    class_names = class_names_from_csv(class_map_path)

    total_time = 0.0
    count = 0

    for file in os.listdir(sound_path):
        if file.endswith('.wav'):
            wav_file_name = sound_path + file
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

            start = time.time()
            scores, embeddings, spectrogram = model(waveform_processed)
            finish = time.time()

            inference_time = finish - start
            total_time += inference_time
            count += 1

            scores_np = scores.numpy()
            infered_class = class_names[scores_np.mean(axis=0).argmax()]
            print(f'path: {sound_path + file}')
            print(f'The main sound is: {infered_class}')

    print("*" * 45)
    print(f'Total time was: {total_time} seconds')
    print(f'Average inference time was: {total_time / count} seconds')
    print(f'the measure was performed on: {count} samples')

    # print(waveform.shape)
    # print(waveform_processed.shape)

    # y = np.expand_dims(waveform, axis=0)

    # print(type(waveform))
    # print(waveform.shape)
    # print(waveform)

    # waveform1 = waveform[:5000]

    # print(waveform1.shape)

    # print(y.shape)
    # print(y)

    # Run the model, check the output.
    # scores, embeddings, spectrogram = model(waveform_processed)
    #
    # scores_np = scores.numpy()
    # infered_class = class_names[scores_np.mean(axis=0).argmax()]
    # print(f'The main sound is: {infered_class}')


def run_yamnet(batch_size, num_of_runs, timeout, sounds_path):
    def run_single_pass(yamnet_model, imagenet):
        shape = (224, 224)
        output = yamnet_model.run()
        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output["densenet169/predictions/Reshape_1:0"][i][0][0]),
                imagenet.extract_top5(output["densenet169/predictions/Reshape_1:0"][i][0][0])
            )

    dataset = Yamnet(batch_size, sounds_path,
                     pre_processing="Yamnet")

    # runner = TFFrozenModelRunner(model_path, ["densenet169/predictions/Reshape_1:0"])

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


# def run_tf_int8(model_path):
#     interpreter = tf.lite.Interpreter(model_path)
#
#     input_details = interpreter.get_input_details()
#     waveform_input_index = input_details[0]['index']
#     output_details = interpreter.get_output_details()
#     scores_output_index = output_details[0]['index']
#     embeddings_output_index = output_details[1]['index']
#     spectrogram_output_index = output_details[2]['index']
#
#     # Input: 3 seconds of silence as mono 16 kHz waveform samples.
#     waveform = np.zeros(3 * 16000, dtype=np.float32)
#
#     # wav_file_name = 'sounds/cough.wav'
#     # print(wav_file_name)
#     # sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
#     # sample_rate, wav_data = ensure_sample_rate(sample_rate, wav_data)
#     #
#     # waveform = wav_data / tf.int16.max
#     #
#     # print(type(waveform))
#
#     interpreter.resize_tensor_input(waveform_input_index, [len(waveform)], strict=True)
#     interpreter.allocate_tensors()
#     interpreter.set_tensor(waveform_input_index, waveform)
#     interpreter.invoke()
#     scores, embeddings, spectrogram = (
#         interpreter.get_tensor(scores_output_index),
#         interpreter.get_tensor(embeddings_output_index),
#         interpreter.get_tensor(spectrogram_output_index))
#
#     print(scores.shape, embeddings.shape, spectrogram.shape)  # (N, 521) (N, 1024) (M, 64)
#
#     # Download the YAMNet class map (see main YAMNet model docs) to yamnet_class_map.csv
#     # See YAMNet TF2 usage sample for class_names_from_csv() definition.
#     class_names = class_names_from_csv(open('/onspecta/dev/knowles_benchmark/yamnet_class_map.csv').read())
#     print(class_names[scores.mean(axis=0).argmax()])  # Should print 'Silence'.


if __name__ == "__main__":
    args = parse_args()
    if args.precision == "fp32":
        run_tf_fp32(args.sound_path, args.intra, args.inter)
    elif args.precision == "int8":
        run_tf_int8(args.model_path)
    elif args.precision == "test":
        run_yamnet()
