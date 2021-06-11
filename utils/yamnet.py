import utils.misc as utils
import utils.dataset as utils_ds
from scipy.io import wavfile
import tensorflow as tf


class Yamnet(utils_ds.AudioDataset):

    def __init__(self, sound_path=None, labels_path=None):

        if sound_path is None:
            env_var = "SOUND_PATH"
            sound_path = utils.get_env_variable(
                env_var, f"Path to audio files directory has not been specified with {env_var} flag")
        if labels_path is None:
            env_var = "AUDIO_LABELS_PATH"
            labels_path = utils.get_env_variable(
                env_var, f"Path to audio files labels has not been specified with {env_var} flag")

        self.sound_path = sound_path
        self.labels_path = labels_path
        self.file_names, self.labels = self.parse_val_file(labels_path)
        self.available_instances = len(self.file_names)
        self.current_sound = 0
        self.correct = 0
        self.ground_truth = None

    def parse_val_file(self, sound_path):
        """
        A function parsing validation file for audioset prepared by OnSpecta.

        .txt file consists of 37 lines each holding data on a single audio file: its file name and 1 label with class
        best describing image's content

        :param sound_path: str, path to file containing audio file names and labels
        :return: list of strings, list of strings
        """

        boundary = 11  # single line of labels file looks like this "sound01.wav Dog"
        with open(sound_path, 'r') as opened_file:
            lines = opened_file.readlines()

        file_names = list()
        labels = list()

        for line in lines:
            file_name = line[:boundary]
            file_names.append(file_name)
            label = line[12:]
            labels.append(label)

        return file_names, labels

    def __get_path_to_audio(self):
        """
        A function providing path to the audio file.

        :return: str, path to the audio file
        """
        try:
            file_name = self.file_names[self.current_sound]
        except IndexError:
            raise utils_ds.OutOfInstances("No more audio files to process in the directory provided")

        self.ground_truth = self.labels[self.current_sound]
        self.current_sound += 1
        return self.sound_path + file_name

    def get_input_array(self):
        """
        A function returning an array containing pre-processed audio file.

        :return: numpy array containing pre-processed audio file requested at class
        initialization
        """

        wav_file_name = self.__get_path_to_audio()
        sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
        sample_rate, wav_data = utils.ensure_sample_rate(sample_rate, wav_data)

        # The wav_data needs to be normalized to values in [-1.0, 1.0]
        waveform = wav_data / tf.int16.max
        # if waveform.shape[0] > 16029:
        #     waveform_processed = waveform[:16029]

        return waveform

    def submit_predictions(self, scores, class_map_path):
        """
        A function meant for submitting a class predictions for a given audio file.

        :param scores: tensorflow.python.framework.ops.EagerTensor, output scores of the model
        :param class_map_path: bytes object, The labels file loaded from the models assets
        :return:
        """
        class_names = utils.class_names_from_csv(class_map_path)
        infered_class = class_names[scores.numpy().mean(axis=0).argmax()]

        self.correct += self.ground_truth.rstrip() == infered_class

    def summarize_accuracy(self):
        """
        A function summarizing the accuracy achieved on the audio files obtained with get_input_array() calls on which
        predictions done where supplied with submit_predictions() function.
        """
        accuracy = self.correct / self.current_sound
        print("\n top 1 accuracy = {:.3f}".format(accuracy))

        print(f"\nAccuracy figures above calculated on the basis of {self.current_sound} audio files.")
        return {"acc": accuracy}
