from utils.helpers import DatasetStub


class StableDiffusion(DatasetStub):

    def __init__(self):
        self._idx = 0
        self.available_instances = 100000

    def get_input(self):
        return 'a professional photograph of an astronaut riding a triceratops'

    def submit_count(self):
        self._idx += 1

    def reset(self):
        self._idx = 0
        return True

    def summarize_accuracy(self):
        print("No accuracy")
        return {"wer_score": None}
