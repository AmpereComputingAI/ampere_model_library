# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
from evaluate import load
from datasets import load_dataset
import utils.misc as utils
from utils.misc import OutOfInstances
from utils.helpers import Dataset


class Covost2(Dataset):
    sampling_rate = 16000

    def __init__(self, dataset_path=None):

        if dataset_path is None:
            env_var = "COMMONVOICE_PATH"
            dataset_path = utils.get_env_variable(
                env_var, f"Path to CommonVoice directory has not been specified with {env_var} flag")

        self._covost2 = load_dataset("covost2", "ja_en", split="validation", data_dir=dataset_path)
        self.available_instances = len(self._covost2["audio"])
        self._idx = 0
        self._translations = []

    def get_input_array(self):
        try:
            return self._covost2["audio"][self._idx]["array"]
        except IndexError:
            raise OutOfInstances

    def submit_translation(self, text: str):
        self._translations.append(text)
        self._idx += 1

    def reset(self):
        self._idx = 0
        self._translations = []
        return True

    def _summarize_accuracy(self):
        assert len(self._translations) == len(self._covost2["translation"][:self._idx])
        bleu_score = load("bleu").compute(
            references=self._covost2["translation"][:self._idx], predictions=self._translations
        )
        return {"bleu_score": bleu_score["bleu"]}
