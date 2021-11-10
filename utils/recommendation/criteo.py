import os
import sys
import torch
import argparse
import numpy as np

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

        self.__data = CriteoDataset(
            dataset="kaggle",
            max_ind_range=-1,
            sub_sample_rate=0.0,
            # max_ind_range=10000000,
            # sub_sample_rate=0.875,
            randomize="total",
            split="test",
            raw_path=dataset_path,
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

        self.__current_id = 0
        self.available_instances = len(self.__data)

    def get_inputs(self):
        """
        A function returning input arrays for DLRM network.
        """
        try:
            for i, val in enumerate(self.__test_loader):
                print(i, val)
                sdf
            dsffsd
            x = [self.__test_loader.collate_fn([self.__data[i] for i in range(self.__current_id, self.__current_id+self.__max_batch_size)])]
            ls_t = list(zip(*x))

            X = torch.cat(ls_t[0])
            (num_s, len_ls) = torch.cat(ls_t[1], dim=1).size()
            lS_o = torch.stack([torch.tensor(range(len_ls)) for _ in range(num_s)])
            lS_i = torch.cat(ls_t[2], dim=1)
            # print(X)
            # print(X.shape)
            # print(lS_o)
            # print(lS_o.shape)
            # print(lS_i)
            # print(lS_i.shape)
            # fsdfsd
            self.__current_id += self.__max_batch_size
            return X, lS_o, lS_i
        except IndexError:
            raise utils.OutOfInstances("No more BraTS19 images to process in the directory provided")

    def summarize_accuracy(self):
        pass
