import utils.misc as utils
import utils.dataset as utils_ds
from scipy.io import wavfile
import tensorflow as tf
import pathlib


class AudioSet(utils_ds.AudioDataset):

    def __init__(self, sound_path=None, labels_path=None):

        if sound_path is None:
            env_var = "SOUND_PATH"
            sound_path = utils.get_env_variable(
                env_var, f"Path to audio files directory has not been specified with {env_var} flag")
        if labels_path is None:
            env_var = "AUDIO_LABELS_PATH"
            labels_path = utils.get_env_variable(
                env_var, f"Path to audio files labels has not been specified with {env_var} flag")

        self.__sound_path = sound_path
        self.__labels_path = labels_path
        self.__file_names, self.__labels = utils.parse_val_file(self.__labels_path, False, 11, True)
        self.__available_instances = len(self.__file_names)
        self.__current_sound = 0
        self.__correct = 0
        self.__ground_truth = None
        super().__init__()

    def __get_path_to_audio(self):
        """
        A function providing path to the audio file.

        :return: str, path to the audio file
        """
        try:
            file_name = self.__file_names[self.__current_sound]
        except IndexError:
            raise utils_ds.OutOfInstances("No more audio files to process in the directory provided")

        self.__ground_truth = self.__labels[self.__current_sound]
        self.__current_sound += 1
        return pathlib.PurePath(self.__sound_path, file_name)

    def get_input_array(self):
        """
        A function returning an array containing pre-processed audio file.

        :return: numpy array containing pre-processed audio file requested at class
        initialization
        """

        wav_file_name = self.__get_path_to_audio()
        sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
        sample_rate, wav_data = self._AudioDataset__ensure_sample_rate(sample_rate, wav_data)

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

        self.__correct += self.__ground_truth.rstrip() == infered_class

    def summarize_accuracy(self):
        """
        A function summarizing the accuracy achieved on the audio files obtained with get_input_array() calls on which
        predictions done where supplied with submit_predictions() function.
        """
        accuracy = self.__correct / self.__current_sound
        print("\n top 1 accuracy = {:.3f}".format(accuracy))

        print(f"\nAccuracy figures above calculated on the basis of {self.__current_sound} audio files.")
        return {"acc": accuracy}
