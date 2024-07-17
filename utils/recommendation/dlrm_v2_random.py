# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC

import torch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

from utils.helpers import Dataset


class RandomDataset(Dataset):
    def __init__(self, batch_size: int) -> None:
        
        self.batch_size = batch_size
        self.__predictions = []
        self.available_instances = 2048000

    def reset(self):
        return False

    def get_inputs(self):
        """
        A function returning input arrays for DLRMv2 network.
        """
        features = torch.rand((self.batch_size, 13))
        sparse_features = KeyedJaggedTensor.from_offsets_sync(keys=[f"cat_{n}" for n in range(26)], values=torch.randint(0, 3, (self.batch_size * 26,)), offsets=torch.tensor([x for x in range(0, self.batch_size * 26 + 1)]),)
        
        return features, sparse_features

    def submit_predictions(self, prediction):
        self.__predictions.append(prediction)

    def _summarize_accuracy(self):
        return
