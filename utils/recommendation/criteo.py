# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import os
import sys
import torch
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_auc_score
import utils.misc as utils
from utils.helpers import Dataset


def append_dlrm_to_pypath():
    dlrm_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dlrm")
    sys.path.append(dlrm_path)


class Criteo(Dataset):
    """
    A class providing facilities for preprocessing of Criteo dataset.
    """

    def __init__(self, max_batch_size, dataset_path=None, debug=False):

        if dataset_path is None:
            env_var = "CRITEO_DATASET_PATH"
            dataset_path = utils.get_env_variable(
                env_var, f"Path to Criteo dataset directory has not been specified with {env_var} flag")

        self.__max_batch_size = max_batch_size

        append_dlrm_to_pypath()
        from utils.recommendation.dlrm.dlrm_data_pytorch import CriteoDataset, collate_wrapper_criteo_offset

        if not debug:
            max_ind_range = 40000000
            self.ln_emb = np.array(
                [39884406, 39043, 17289, 7420, 20263, 3, 7120, 1543, 63, 38532951, 2953546, 403346, 10, 2208, 11938,
                 155, 4,
                 976, 14, 39979771, 25641295, 39664984, 585935, 12972, 108, 36])
            sub_sample_rate = 0.0
        else:
            max_ind_range = 10000000
            self.ln_emb = np.array(
                [9980333, 36084, 17217, 7378, 20134, 3, 7112, 1442, 61, 9758201, 1333352, 313829, 10, 2208, 11156, 122,
                 4,
                 970, 14, 9994222, 7267859, 9946608, 415421, 12420, 101, 36])
            sub_sample_rate = 0.875

        self.__data = CriteoDataset(
            dataset="terabyte",
            max_ind_range=max_ind_range,
            sub_sample_rate=sub_sample_rate,
            randomize="total",
            split="test",
            raw_path=str(Path(dataset_path, "day")),
            pro_data=str(Path(dataset_path, "terabyte_processed.npz")),
            memory_map=True,
            dataset_multiprocessing=True
        )

        self.__test_loader = torch.utils.data.DataLoader(
            self.__data,
            batch_size=self.__max_batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collate_wrapper_criteo_offset,
            pin_memory=False,
            drop_last=False
        )

        self.available_instances = len(self.__test_loader) * self.__test_loader.batch_size
        self.dataset_iterator = self._generate_input()
        self.correct_count = 0
        self.total_count = 0
        self.true_positives = 0
        self.false_positives = 0
        self.false_negatives = 0

        self.predictions = []
        self.targets = []

    def reset(self):
        return False

    def _generate_input(self):
        for val in self.__test_loader:
            yield val

    def get_inputs(self):
        """
        A function returning input arrays for DLRM network.
        """
        try:
            val = next(self.dataset_iterator)
        except StopIteration:
            raise utils.OutOfInstances("No more Criteo samples to process in the directory provided")
        self.__single_input = val[0], val[1], val[2]
        self.__labels = val[3]
        return self.__single_input

    def submit_predictions(self, prediction):
        prediction = prediction >= 0.5
        result = prediction == self.__labels

        self.predictions.append(prediction)
        self.targets.append(self.__labels)

        self.total_count += len(prediction)
        self.correct_count += sum(result)
        self.true_positives += sum(torch.logical_and(prediction == 1, self.__labels == 1))
        self.false_positives += sum(torch.logical_and(prediction == 1, self.__labels == 0))
        self.false_negatives += sum(torch.logical_and(prediction == 0, self.__labels == 1))

    def summarize_accuracy(self):
        accuracy = self.correct_count / self.total_count
        precision = self.true_positives / (self.true_positives + self.false_positives)
        recall = self.true_positives / (self.true_positives + self.false_negatives)
        auc = roc_auc_score(np.concatenate(self.predictions), np.concatenate(self.targets))

        # print("\n Accuracy = {:.3f}".format(accuracy.item()))
        # print("\n Precision = {:.3f}".format(precision.item()))
        # print("\n Recall = {:.3f}".format(recall.item()))
        # print("\n AUC = {:.3f}".format(auc))
        # print(f"\nAccuracy figures above calculated on the basis of {self.total_count} samples.")
        return {"accuracy": accuracy, "precision": precision, "recall": recall, "auc": auc}
