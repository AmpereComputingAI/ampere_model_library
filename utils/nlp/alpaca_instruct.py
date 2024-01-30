from collections import Counter
import json

import utils.misc as utils

class AlpacaInstruct:
    """
    A class providing facilities for preprocessing and postprocessing of Alpaca dataset.
    """

    def __init__(self, batch_size: int, tokenize_func=None, detokenize_func=None, dataset_path=None):
        if dataset_path is None:
            env_var = "ALPACA_DATASET_PATH"
            dataset_path = utils.get_env_variable(
                env_var, f"Path to Alpaca dataset has not been specified with {env_var} flag")

        with open(dataset_path, "r") as dataset_file:
            data = dataset_file.read()
            self.data = json.loads(data)

        self.available_instances = len(self.data)
        self.__current_sample = 0
        self.__exact_match_count = 0
        self.__f1_count = 0

    @staticmethod
    def preprocess(data):
        """
        A function converting the raw input data into a format expected by Alpaca.
        """

        prompt = ("Below is an instruction that describes a task. "
                  "Write a response that appropriately completes the request.\r\n\r\n"
                  "### Instruction:\r\n"
                  f"{data['instruction']}\r\n\r\n")
        if data['input']:
            prompt += ("### Input:\r\n"
                       f"{data['input']}\r\n\r\n")
        prompt += "### Response:"

        return prompt

    def get_input_array(self):
        return self.data[self.__current_sample]

    def reset(self):
        self.__current_sample = 0
        return True
    
    def submit_prediction(self, answer: str):
        """
        A function allowing for a submission of obtained results of NLP inference.

        :param answer: string, detokenized answer
        """

        def f1_score(normalized_prediction, normalized_ground_truth):
            """
            A function calculating the F1 score betweed normalized prediction and normalized ground truth.

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

        def metric_max_over_ground_truth(metric_fn, prediction, ground_truth):
            """
            A function applying given metric function over provided correct answer (ground_truth).

            :param metric_fn: function calculating a metric
            :param prediction: str with predicted answer
            :param ground_truth: string of correct answer

            :return: float, max score obtained
            """
            scores_for_ground_truths = []
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
            return max(scores_for_ground_truths)

        ground_truth = self.data[self.__current_sample]['output']
        self.__exact_match_count += metric_max_over_ground_truth(exact_match_score, answer, ground_truth)
        self.__f1_count += metric_max_over_ground_truth(f1_score, answer, ground_truth)
        self.__current_sample += 1
    
    def summarize_accuracy(self):
        exact_match = self.__exact_match_count / self.__current_sample
        print("\n Exact match = {:.3f}".format(exact_match))
        f1 = self.__f1_count / self.__current_sample
        print(" F1 = {:.3f}".format(f1))

        print(f"\nAccuracy figures above calculated on the basis of {self.__current_sample} instructions processed.")
        return {"exact_match": exact_match, "f1": f1}
