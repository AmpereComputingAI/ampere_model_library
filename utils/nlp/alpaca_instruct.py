# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
from collections import Counter
import json
import utils.misc as utils
from utils.helpers import Dataset


class AlpacaInstruct(Dataset):
    """
    A class providing facilities for preprocessing and postprocessing of Alpaca dataset.
    """

    def __init__(self, batch_size: int, dataset_path=None):
        self._batch_size = batch_size

        if dataset_path is None:
            env_var = "ALPACA_DATASET_PATH"
            dataset_path = utils.get_env_variable(
                env_var, f"Path to Alpaca dataset has not been specified with {env_var} flag")

        with open(dataset_path, "r") as dataset_file:
            data = dataset_file.read()
            self.data = json.loads(data)

        self.available_instances = len(self.data)
        self._current_sample = -1
        self._count = 0
        self._exact_match = 0
        self._f1 = 0

    def get_input_string(self):
        self._current_sample += 1
        assert self._current_sample * self._batch_size == self._count

        prompt = ("Below is an instruction that describes a task. "
                  "Write a response that appropriately completes the request.\r\n\r\n"
                  f"### Instruction:\r\n{self.data[self._current_sample]['instruction']}\r\n\r\n")
        if self.data[self._current_sample]['input']:
            prompt += f"### Input:\r\n{self.data[self._current_sample]['input']}\r\n\r\n"
        prompt += "### Response:"

        return prompt

    def reset(self):
        self._current_sample = 0
        return True

    def submit_prediction(self, answer: str):
        """
        A function allowing for a submission of obtained results of NLP inference.

        :param answer: string, detokenized answer
        """

        def f1_score(normalized_prediction, normalized_ground_truth):
            """
            A function calculating the F1 score between normalized prediction and normalized ground truth.

            :param normalized_prediction: str, normalized answer (prediction)
            :param normalized_ground_truth: str, normalized correct answer (gt)
            :return: float, f1 score
            """
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                return 0
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1

        def exact_match_score(normalized_prediction, normalized_ground_truth):
            """
            A function comparing normalized strings.

            :param normalized_prediction: str, normalized answer (prediction)
            :param normalized_ground_truth: str, normalized correct answer (gt)
            :return: bool, True if strings equal
            """
            return normalized_prediction == normalized_ground_truth

        def metric_max_over_ground_truth(metric_fn, pred, gt):
            """
            A function applying given metric function over provided correct answer (ground_truth).

            :param metric_fn: function calculating a metric
            :param pred: str with predicted answer
            :param gt: string of correct answer

            :return: float, max score obtained
            """
            scores_for_ground_truths = []
            score = metric_fn(pred, gt)
            scores_for_ground_truths.append(score)
            return max(scores_for_ground_truths)

        ground_truth = self.data[self._current_sample]['output']
        self._exact_match += metric_max_over_ground_truth(exact_match_score, answer, ground_truth)
        self._f1 += metric_max_over_ground_truth(f1_score, answer, ground_truth)
        self._count += 1

    def summarize_accuracy(self):
        exact_match = self._exact_match / self._count
        f1 = self._f1 / self._count
        return {"exact_match": exact_match, "f1": f1}
