# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import numpy as np
import json
import re
import string
from collections import Counter
import pathlib
import utils.misc as utils


class Squad_v1_1:
    """
    A class providing facilities for preprocessing and postprocessing of Squad v1.1 validation dataset.
    """

    def __init__(self, batch_size: int, tokenize_func, detokenize_func, target_seq_size=None, dataset_path=None):

        if dataset_path is None:
            env_var = "SQUAD_V1_1_PATH"
            dataset_path = utils.get_env_variable(
                env_var, f"Path to Squad v1.1 .json file has not been specified with {env_var} flag")

        self.__batch_size = batch_size
        self.__target_seq_size = target_seq_size
        self.__tokenize_func = tokenize_func
        self.__detokenize_func = detokenize_func
        self.__dataset = self.__verify_and_load(dataset_path)
        self.__example_iterator = self.__examples()

        self.__questions_count = 0
        self.__unanswered_questions_count = 0
        self.available_instances = self.__get_num_questions()
        self.__current_inputs = None
        self.__exact_match_count = 0
        self.__f1_count = 0

    def __get_num_questions(self):
        """
        A function calculating number of available examples (questions) in dataset file provided.

        :return: int, number of questions available
        """
        total_questions = 0
        for section in self.__dataset:
            for paragraph in section["paragraphs"]:
                total_questions += len(paragraph["qas"])
        return total_questions

    def __verify_and_load(self, dataset_path, expected_version="1.1"):
        """
        A function loading .json file of Squad v1.1 validation dataset and verifying its version.

        :param dataset_path: str, path to file containing validation data (contextes, questions, possible answers)
        :param expected_version: str, version of Squad dataset expected
        :return: dictionary with nested dictionaries / lists, dataset
        """
        with open(dataset_path) as dataset_file:
            dataset_json = json.load(dataset_file)
            encountered_version = dataset_json["version"]
            if encountered_version != expected_version:
                print_goodbye_message_and_die(
                    f"Expected SQUAD version {expected_version} but encountered version {encountered_version}")
            dataset = dataset_json["data"]
        return dataset

    def __examples(self):
        """
        A generator of examples iterating over the dataset with per question stepping.

        :yield: str, str, list: context, questions, list of possible (correct) answers
        """
        for section in self.__dataset:
            for paragraph in section["paragraphs"]:
                for qas in paragraph["qas"]:
                    yield paragraph["context"], qas["question"], qas["answers"]

    def __load_next_inputs_maybe(self):
        """
        A function that loads new examples in the quantity equal to the requested batch size under the condition that
        previously issued questions have already been answered.
        """
        if self.__unanswered_questions_count == 0:
            contextes = list()
            questions = list()
            self.__valid_answers = list()
            for _ in range(self.__batch_size):
                try:
                    context, question, correct_answers = next(self.__example_iterator)
                except StopIteration:
                    raise utils.OutOfInstances("No more examples to process in the Squad file provided.")
                contextes.append(context)
                questions.append(question)
                self.__questions_count += 1
                self.__unanswered_questions_count += 1
                self.__valid_answers.append(correct_answers)
            self.__current_inputs = self.__tokenize_func(questions, contextes)

    def __get_input_array(self, input_name: string):
        """
        A function filling numpy arrays of proper batch size with examples.
        Padding is applied if requested / necessary.
        Cropping is applied when necessary.

        :param input_name: str, input name as used by tokenization function
        :return: numpy array with raw/padded/cropped input
        """
        self.__load_next_inputs_maybe()
        input = self.__current_inputs[input_name]

        # if seq size has not been specified the target one will be set to the size of the longest sequence in a batch
        if self.__target_seq_size is None:
            target_seq_size = 0
            for i in range(self.__batch_size):
                target_seq_size = max(len(input[i]), target_seq_size)
        else:
            target_seq_size = self.__target_seq_size

        input_padded = np.empty([self.__batch_size, target_seq_size])

        for i in range(self.__batch_size):
            length_of_seq = len(input[i])
            if target_seq_size >= length_of_seq:
                # padding is applied when target seq size is longer than or equal to encountered size
                space_to_pad = target_seq_size - length_of_seq
                input_padded[i] = np.pad(input[i], (0, space_to_pad), "constant", constant_values=0)
            else:
                # cropping is applied if otherwise
                # TODO: this should actually be caused by exceeding max_seq_size, not target_seq_size
                input_padded[i] = input[i][0:target_seq_size]

        return input_padded

    def reset(self):
        self.__example_iterator = self.__examples()
        self.__questions_count = 0
        self.__unanswered_questions_count = 0
        self.__current_inputs = None
        self.__exact_match_count = 0
        self.__f1_count = 0
        return True

    def get_input_arrays(self):
        self.__load_next_inputs_maybe()
        return self.__current_inputs

    def get_input_ids_array(self):
        return self.__get_input_array("input_ids")

    def get_attention_mask_array(self):
        return self.__get_input_array("attention_mask")

    def get_token_type_ids_array(self):
        return self.__get_input_array("token_type_ids")

    def extract_answer(self, id_in_batch: int, answer_start_id: int, answer_end_id: int):
        """
        A function extracting the answering part of context and applying the provided detokenization function on it.

        :param id_in_batch: int, index in input batch that answer relates to
        :param answer_start_id: int, index of answer start in context text
        :param answer_end_id: int, index of answer end in context text
        :return: string, detokenized answer
        """
        answer = self.__current_inputs["input_ids"][id_in_batch][answer_start_id:answer_end_id+1]
        return self.__detokenize_func(answer)

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
                score = metric_fn(normalize(prediction), normalize(ground_truth["text"]))
                scores_for_ground_truths.append(score)
            return max(scores_for_ground_truths)

        ground_truths = self.__valid_answers[id_in_batch]
        self.__exact_match_count += metric_max_over_ground_truths(exact_match_score, answer, ground_truths)
        self.__f1_count += metric_max_over_ground_truths(f1_score, answer, ground_truths)
        self.__unanswered_questions_count -= 1

    def summarize_accuracy(self):
        """
        A function summarizing the accuracy achieved on the questions obtained with get_*_array() calls on which
        predicted answers were supplied with submit_predictions() function.

        :return: dict, dictionary containing two metrics produced: exact_match signifying the ratio of perfect answers
        and f1 metric
        """
        if self.__unanswered_questions_count != 0:
            utils.print_goodbye_message_and_die(
                "Answers for some of the issued questions have not been submitted.")

        exact_match = self.__exact_match_count / self.__questions_count
        print("\n Exact match = {:.3f}".format(exact_match))

        f1 = self.__f1_count / self.__questions_count
        print(" F1 = {:.3f}".format(f1))

        print(f"\nAccuracy figures above calculated on the basis of {self.__questions_count} questions answered.")
        return {"exact_match": exact_match, "f1": f1}
