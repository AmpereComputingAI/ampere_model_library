import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path

import utils.misc as utils


def append_dlrm_to_pypath():
    dlrm_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dlrm")
    sys.path.append(dlrm_path)


class Criteo:
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
                [39884406, 39043, 17289, 7420, 20263, 3, 7120, 1543, 63, 38532951, 2953546, 403346, 10, 2208, 11938, 155, 4,
                976, 14, 39979771, 25641295, 39664984, 585935, 12972, 108, 36])
            sub_sample_rate = 0.0
        else:
            max_ind_range = 10000000
            self.ln_emb = np.array(
                [9980333, 36084, 17217, 7378, 20134, 3, 7112, 1442, 61, 9758201, 1333352, 313829, 10, 2208, 11156, 122, 4,
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

        for val in self.__test_loader:
            self.__single_input = val[0], val[1], val[2]

        self.available_instances = 1

    def reset(self):
        return False

    def get_inputs(self):
        """
        A function returning input arrays for DLRM network.
        """
        return self.__single_input

    def summarize_accuracy(self):
        pass
