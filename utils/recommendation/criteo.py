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
    A class providing facilities for preprocessing and postprocessing of Criteo Kaggle competition dataset.
    """

    def __init__(self, max_batch_size=2048, dataset_path=None):

        if dataset_path is None:
            env_var = "CRITEO_DATASET_PATH"
            dataset_path = utils.get_env_variable(
                env_var, f"Path to Criteo Kaggle dataset directory has not been specified with {env_var} flag")

        self.__max_batch_size = max_batch_size

        append_dlrm_to_pypath()
        from utils.recommendation.dlrm.dlrm_data_pytorch import CriteoDataset, collate_wrapper_criteo_offset

        self.ln_emb = np.array(
            [39884406, 39043, 17289, 7420, 20263, 3, 7120, 1543, 63, 38532951, 2953546, 403346, 10, 2208, 11938, 155, 4,
             976, 14, 39979771, 25641295, 39664984, 585935, 12972, 108, 36]
        )

        self.__data = CriteoDataset(
            dataset="terabyte",
            max_ind_range=40000000,
            sub_sample_rate=0.0,
            # max_ind_range=10000000,
            # sub_sample_rate=0.875,
            randomize="total",
            split="test",
            raw_path=Path(dataset_path, "day"),
            pro_data=Path(dataset_path, "terabyte_processed.npz"),
            # memory_map=True,
            memory_map=True,
            dataset_multiprocessing=True
        )

        self.__test_loader = torch.utils.data.DataLoader(
            self.__data,
            batch_size=self.__max_batch_size,
            shuffle=False,
            num_workers=0,  # 16, # 0
            collate_fn=collate_wrapper_criteo_offset,
            pin_memory=False,
            drop_last=False
        )

        self.__current_id = 0
        self.available_instances = len(self.__data)

    def __input_batch(self):
        try:
            for val in self.__test_loader:
                yield val[0], val[1], val[2]
        except IndexError:
            raise utils.OutOfInstances("No more BraTS19 images to process in the directory provided")

    def get_inputs(self):
        """
        A function returning input arrays for DLRM network.
        """
        return next(self.__input_batch())

    def summarize_accuracy(self):
        pass
