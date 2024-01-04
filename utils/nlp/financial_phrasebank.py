from utils.helpers import DatasetStub


class FinancialPhraseBank(DatasetStub):
    def __init__(self):
        self.available_instances = float("inf")
        self.string = "Stocks rallied and the British pound gained. " * 100

    def get_input_string(self):
        return self.string

    def submit_prediction(self, predictions):
        pass

    def summarize_accuracy(self) -> dict:
        return {}
