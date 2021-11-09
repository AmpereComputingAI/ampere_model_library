import os
import sys
import torch
import argparse
import numpy as np


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

        self.__data = CriteoDataset(
            dataset="kaggle",
            max_ind_range=-1,
            sub_sample_rate=0.0,
            randomize="total",
            split="test",
            raw_path=dataset_path,
            memory_map=True
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

        self.__current_id = 0
        self.available_instances = len(self.__data)

    def get_inputs(self):
        """
        A function returning input arrays for DLRM network.
        """
        try:
            print(self.__test_loader.collate_fn([self.__data[i] for i in range(self.__current_id, self.__max_batch_size)]))
            sdsd
        except IndexError:
            raise utils.OutOfInstances("No more BraTS19 images to process in the directory provided")
        self.__input_details.append(data[1])
        return data[0]
