# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import re
import string
from collections import Counter
import utils.misc as utils
from utils.helpers import Dataset


class Lambada(Dataset):
    """
    A class providing facilities for preprocessing and postprocessing of LAMBADA dataset.
    """

    def __init__(self, batch_size: int, tokenize_func, detokenize_func, dataset_path=None):
        pass

        if dataset_path is None:
            env_var = "LAMBADA_PATH"
            dataset_path = utils.get_env_variable(
                env_var, f"Path to LAMBADA dataset has not been specified with {env_var} flag")

        self.__batch_size = batch_size
        self.__tokenize_func = tokenize_func
        self.__detokenize_func = detokenize_func
        self.__dataset = dataset_path
        self.__example_iterator = self.__examples()

        self.__texts_count = 0
        self.__unsubmitted_count = 0
        self.available_instances = sum(1 for _ in self.__examples())
        self.__current_inputs = None
        self.__exact_match_count = 0
        self.__f1_count = 0

    def __examples(self):
        with open(self.__dataset, encoding="utf-8") as f:
            for line in f:
                yield line

    def __load_next_inputs_maybe(self):
        """
        A function that loads new examples in the quantity equal to the requested batch size under the condition that
        previously issued questions have already been answered.
        """
        if self.__unsubmitted_count == 0:
            lines = []
            self.__last_words = []
            examples_needed = self.__batch_size
            while examples_needed > 0:
                try:
                    line, last_word = next(self.__example_iterator).rsplit(" ", 1)
                except StopIteration:
                    raise utils.OutOfInstances("No more examples to process in the LAMBADA file provided.")

                examples_needed -= 1
                lines.append(self.__tokenize_func(line))
                self.__last_words.append(last_word)
                self.__texts_count += 1
                self.__unsubmitted_count += 1

            self.__current_inputs = lines

    def get_input_array(self):
        self.__load_next_inputs_maybe()
        input = self.__current_inputs
        return input

    def reset(self):
        self.__example_iterator = self.__examples()
        self.__texts_count = 0
        self.__unsubmitted_count = 0
        self.__current_inputs = None
        return True

    def submit_prediction(self, id_in_batch: int, answer: string):
        """
        A function allowing for a submission of obtained results of NLP inference.

        :param id_in_batch: int, index in input batch that answer relates to
        :param answer: string, detokenized answer
        """

        def normalize(answer_string):
            """
            A function normalizing the answer to mitigate the effect of formatting.

            :param answer_string: str, answer
            :return: str, normalized answer
            """

            def remove_articles(text):
                return re.sub(r'\b(a|an|the)\b', ' ', text)

            def white_space_fix(text):
                return ' '.join(text.split())

            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)

            def lower(text):
                return text.lower()

            return white_space_fix(remove_articles(remove_punc(lower(answer_string))))

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

        def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
            """
            A function applying given metric function over provided acceptable answers (ground_truths).

            :param metric_fn: function calculating a metric
            :param prediction: str with predicted answer
            :param ground_truths: list of strings, list of acceptable answers

            :return: float, max score obtained
            """
            scores_for_ground_truths = []
            for ground_truth in ground_truths:
                # score = metric_fn(prediction.lower().strip(), ground_truth.lower().strip())
                score = metric_fn(normalize(prediction), normalize(ground_truth))

                scores_for_ground_truths.append(score)
            return max(scores_for_ground_truths)

        ground_truths = [self.__last_words[id_in_batch]]
        self.__exact_match_count += metric_max_over_ground_truths(exact_match_score, answer, ground_truths)
        self.__f1_count += metric_max_over_ground_truths(f1_score, answer, ground_truths)
        self.__unsubmitted_count -= 1

    def _summarize_accuracy(self):
        """
        A function summarizing the accuracy achieved on the questions obtained with get_*_array() calls on which
        predicted answers were supplied with submit_predictions() function.

        :return: dict, dictionary containing two metrics produced: exact_match signifying the ratio of perfect answers
        and f1 metric
        """
        if self.__unsubmitted_count != 0:
            utils.print_goodbye_message_and_die(
                "Answers for some of the issued questions have not been submitted.")

        exact_match = self.__exact_match_count / self.__texts_count
        # print("\n Exact match = {:.3f}".format(exact_match))

        f1 = self.__f1_count / self.__texts_count
        # print(" F1 = {:.3f}".format(f1))

        # print(f"\nAccuracy figures above calculated on the basis of {self.__texts_count} questions answered.")
        return {"exact_match": exact_match, "f1": f1}
