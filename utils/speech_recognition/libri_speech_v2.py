# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import os
from evaluate import load
from datasets import load_dataset
from utils.misc import OutOfInstances
from utils.helpers import Dataset


class LibriSpeech(Dataset):
    sampling_rate = 16000

    def __init__(self):
        self._librispeech = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        self.available_instances = len(self._librispeech["audio"])
        self._idx = 0
        self._transcriptions = []

    def get_input_array(self):
        try:
            return self._librispeech["audio"][self._idx]["array"]
        except IndexError:
            if os.environ.get("IGNORE_DATASET_LIMITS") == "1":
                if self.reset():
                    return self.get_input_array()
            raise OutOfInstances

    def submit_transcription(self, text: str):
        if self.do_skip():
            return

        self._transcriptions.append(text)
        self._idx += 1

    def reset(self):
        self._idx = 0
        self._transcriptions = []
        return True

    def summarize_accuracy(self):
        if self.do_skip():
            return {}

        assert len(self._transcriptions) == len(self._librispeech["text"][:self._idx])
        wer_score = load("wer").compute(
            references=self._librispeech["text"][:self._idx], predictions=self._transcriptions
        )
        # print("\n  WER score = {:.3f}".format(wer_score))
        # print(f"\n  Accuracy figures above calculated on the basis of {self._idx} sample(s).")
        return {"wer_score": wer_score}
