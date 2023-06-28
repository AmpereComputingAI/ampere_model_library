from evaluate import load
from datasets import load_dataset
from utils.misc import OutOfInstances
from utils.helpers import DatasetStub


class LibriSpeech(DatasetStub):
    def __init__(self):
        self._librispeech = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        self._idx = 0
        self._transcriptions = {}

    def get_input_array(self):
        try:
            return self._librispeech["audio"][self._idx]["array"]
        except IndexError:
            raise OutOfInstances

    def submit_transcription(self, text: str):
        self._transcriptions[self._idx] = text
        self._idx += 1

    def reset(self):
        self._idx = 0
        self._transcriptions = {}
        return True

    def summarize_accuracy(self):
        assert len(self._transcriptions) == len(self._librispeech["text"][:self._idx])
        wer_score = load("wer").compute(
            references=self._librispeech["text"][:self._idx], predictions=self._transcriptions
        )
        print("\n WER score = {:.3f}".format(wer_score))
        print(f"\nAccuracy figures above calculated on the basis of {self._idx} samples.")
        return {"wer_score": wer_score}
