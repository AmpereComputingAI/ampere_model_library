import utils.misc as utils
import utils.dataset as utils_ds
from scipy.io import wavfile
import tensorflow as tf


class Yamnet(utils_ds.AudioDataset):

    def __init__(self, batch_size: int, sound_path=None, labels_path=None):

        if sound_path is None:
            env_var = "SOUND_PATH"
            sound_path = utils.get_env_variable(
                env_var, f"Path to ImageNet images directory has not been specified with {env_var} flag")

        self.sound_path = sound_path
        self.labels_path = labels_path
        self.batch_size = batch_size
        self.file_names, self.labels = self.parse_val_file(labels_path)
        self.current_sound = 0
        self.correct = 0

        print(self.file_names)
        print(self.labels)

    def parse_val_file(self, sound_path):
        """
        A function parsing validation file for ImageNet 2012 validation dataset.

        .txt file consists of 50000 lines each holding data on a single image: its file name and 1 label with class best
        describing image's content

        :param labels_path: str, path to file containing image file names and labels
        :param is1001classes: bool, parameter setting whether the tested model has 1001 classes (+ background) or
        original 1000 classes
        :return: list of strings, list of ints
        """

        boundary = 11  # single line of labels file looks like this "sound01.wav 456"
        with open(sound_path, 'r') as opened_file:
            lines = opened_file.readlines()

        file_names = list()
        labels = list()

        for line in lines:
            file_name = line[:boundary]
            file_names.append(file_name)
            label = line[boundary:]
            labels.append(label)

        return file_names, labels

    def __get_path_to_audio(self):
        print(self.current_sound)
        try:
            file_name = self.file_names[self.current_sound]
        except IndexError:
            raise utils_ds.OutOfInstances("No more ImageNet images to process in the directory provided")

        self.current_sound += 1
        print(self.sound_path + file_name)
        return self.sound_path + file_name

    def get_input_array(self):

        wav_file_name = self.__get_path_to_audio()
        sample_rate, wav_data = wavfile.read(wav_file_name, 'rb')
        sample_rate, wav_data = utils.ensure_sample_rate(sample_rate, wav_data)

        # The wav_data needs to be normalized to values in [-1.0, 1.0]
        waveform = wav_data / tf.int16.max

        if waveform.shape[0] > 16029:
            waveform_processed = waveform[:16029]

        return waveform_processed

    def submit_predictions(self, scores, class_map_path):
        try:
            ground_truth = self.labels[self.current_sound]
        except IndexError:
            raise utils_ds.OutOfInstances("No more sounds to process in the directory provided")

        class_names = utils.class_names_from_csv(class_map_path)
        scores_np = scores.numpy()
        infered_class = class_names[scores_np.mean(axis=0).argmax()]
        self.correct += int(ground_truth == infered_class)

    def summarize_accuracy(self):
        accuracy = self.correct / self.current_sound
        print("\n Top-1 accuracy = {:.3f}".format(accuracy))

        print(f"\nAccuracy figures above calculated on the basis of {self.current_sound} images.")
        return {"top_1_acc": accuracy}
