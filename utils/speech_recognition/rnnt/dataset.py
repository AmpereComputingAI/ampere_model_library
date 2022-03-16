# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains classes and functions related to data loading
"""
from collections import namedtuple
import torch
import numpy as np
from torch.utils.data import Dataset

from .manifest import Manifest

class AudioDataset(Dataset):
    def __init__(self, dataset_dir, manifest_filepath, labels, featurizer, max_duration=None, pad_to_max=False,
                 min_duration=None, blank_index=0, max_utts=0, normalize=True, sort_by_duration=False,
                 trim=False, speed_perturbation=False):
        """Dataset that loads tensors via a json file containing paths to audio files, transcripts, and durations
        (in seconds). Each entry is a different audio sample.
        Args:
            dataset_dir: absolute path to dataset folder
            manifest_filepath: relative path from dataset folder to manifest json as described above.
            labels: String containing all the possible characters to map to
            featurizer: Initialized featurizer class that converts paths of audio to feature tensors
            max_duration: If audio exceeds this length, do not include in dataset
            min_duration: If audio is less than this length, do not include in dataset
            pad_to_max: if specified input sequences into dnn model will be padded to max_duration
            blank_index: blank index for ctc loss / decoder
            max_utts: Limit number of utterances
            normalize: whether to normalize transcript text
            sort_by_duration: whether or not to sort sequences by increasing duration
            trim: if specified trims leading and trailing silence from an audio signal.
            speed_perturbation: specify if using data contains speed perburbation
        """
        m_paths = [manifest_filepath]
        self.manifest = Manifest(dataset_dir, m_paths, labels, blank_index, pad_to_max=pad_to_max,
                                 max_duration=max_duration,
                                 sort_by_duration=sort_by_duration,
                                 min_duration=min_duration, max_utts=max_utts,
                                 normalize=normalize, speed_perturbation=speed_perturbation)
        self.featurizer = featurizer
        self.blank_index = blank_index
        self.trim = trim
        print(
            "Dataset loaded with {0:.2f} hours. Filtered {1:.2f} hours.".format(
                self.manifest.duration / 3600,
                self.manifest.filtered_duration / 3600))

    def __getitem__(self, index):
        sample = self.manifest[index]
        rn_indx = np.random.randint(len(sample['audio_filepath']))
        duration = sample['audio_duration'][rn_indx] if 'audio_duration' in sample else 0
        offset = sample['offset'] if 'offset' in sample else 0
        features = self.featurizer.process(sample['audio_filepath'][rn_indx],
                                           offset=offset, duration=duration,
                                           trim=self.trim)

        AudioSample = namedtuple('AudioSample', ['waveform',
                                                 'transcript'])
        return AudioSample(features,
                           torch.LongTensor(sample["transcript"]))

    def __len__(self):
        return len(self.manifest)
