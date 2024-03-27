# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC

import pathlib

import torch

import utils.misc as utils
import utils.speech_recognition.rnnt.dataset as utils_dataset
from utils.speech_recognition.rnnt.features import WaveformFeaturizer
from utils.speech_recognition.rnnt.preprocessing import AudioPreprocessing
from utils.speech_recognition.rnnt.metrics import word_error_rate
from utils.helpers import Dataset


class LibriSpeech(Dataset):

    def __init__(self, dataset_dir_path, config, max_batch_size):

        if dataset_dir_path is None:
            env_var = "LIBRISPEECH_DATASET_PATH"
            dataset_dir_path = utils.get_env_variable(
                env_var, f"Path to LibriSpeech dataset directory has not been specified with {env_var} flag")

        self.__featurizer = WaveformFeaturizer.from_config(
            config['input_eval'], perturbation_configs=config["input"])
        self.__labels = config["labels"]["labels"]
        self.__data = utils_dataset.AudioDataset(
            dataset_dir_path,
            pathlib.Path(dataset_dir_path) / 'dev-clean-wav.json',
            self.__labels,
            self.__featurizer,
            blank_index=27, )
        self.__dataloader = iter(torch.utils.data.DataLoader(
            dataset=self.__data,
            batch_size=max_batch_size,
            collate_fn=lambda b: self.seq_collate_fn(b),
            drop_last=False,
            shuffle=True,
            num_workers=0,
            pin_memory=True,
            sampler=None
        ))
        self.__preprocessor = AudioPreprocessing(**config["input_eval"])
        self.__current_input = None
        self.__word_count = 0
        self.__score_count = 0

    @staticmethod
    def seq_collate_fn(batch):
        """batches samples and returns as tensors
        Args:
        batch : list of samples
        Returns
        batches of tensors
        """
        audio_lengths = torch.LongTensor([sample.waveform.size(0)
                                          for sample in batch])
        transcript_lengths = torch.LongTensor([sample.transcript.size(0)
                                               for sample in batch])
        permute_indices = torch.argsort(audio_lengths, descending=True)

        audio_lengths = audio_lengths[permute_indices]
        transcript_lengths = transcript_lengths[permute_indices]
        padded_audio_signals = torch.nn.utils.rnn.pad_sequence(
            [batch[i].waveform for i in permute_indices],
            batch_first=True
        )
        transcript_list = [batch[i].transcript
                           for i in permute_indices]
        packed_transcripts = torch.nn.utils.rnn.pack_sequence(transcript_list,
                                                              enforce_sorted=False)

        return (padded_audio_signals, audio_lengths, transcript_list,
                packed_transcripts, transcript_lengths)

    def get_input_arrays(self):
        try:
            input = next(self.__dataloader)
        except StopIteration:
            raise utils.OutOfInstances("No more LibriSpeech sequences to process in the directory provided")
        self.__current_input = input
        feature, feature_length = self.__preprocessor.forward((input[0], input[1]))
        feature = feature.permute(2, 0, 1)
        return feature, feature_length

    def __decode(self, x):
        return "".join([self.__labels[letter] for letter in x])

    def submit_predictions(self, predictions):
        """
        A function meant for submitting a prediction for a given sequence.

        """
        reference = [self.__decode(x) for x in self.__current_input[2]]
        predictions = [self.__decode(x) for x in predictions]

        _, scores, words = word_error_rate(predictions, reference)
        self.__score_count += scores
        self.__word_count += words

    def summarize_accuracy(self):
        """
        A function summarizing the accuracy achieved on the sequences obtained with get_input_array() calls on which
        predictions done where supplied with submit_predictions() function.
        """
        if self.__word_count != 0:
            accuracy = 1 - ((1.0 * self.__score_count) / self.__word_count)
        else:
            accuracy = 0.0

        # print("Accuracy = {:.3f}".format(accuracy))
        # print(f"\nAccuracy figures above calculated on the basis of {self.__word_count} words.")
        return {"accuracy": accuracy}
