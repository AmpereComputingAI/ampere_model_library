# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import json
import re
import string
from collections import Counter

from sacrebleu.metrics import BLEU

import utils.misc as utils


class WMT:
    """
    A class providing facilities for preprocessing and postprocessing of WMT validation dataset.
    """

    def __init__(self, batch_size: int, dataset_path=None, targets_path=None):

        if dataset_path is None:
            env_var = "WMT_DATASET_PATH"
            dataset_path = utils.get_env_variable(
                env_var, f"Path to WMT file has not been specified with {env_var} flag")
        if targets_path is None:
            env_var = "WMT_TARGETS_PATH"
            targets_path = utils.get_env_variable(
                env_var, f"Path to WMT targets files has not been specified with {env_var} flag")
        
        self.dataset_path = dataset_path
        self.targets_path = targets_path

        self.__batch_size = batch_size
        self.__example_iterator = self.__examples()

        self.__questions_count = 0
        self.__unanswered_questions_count = 0
        self.available_instances = self.__get_num_sentences()
        self.__current_inputs = None
        self.__exact_match_count = 0
        self.__f1_count = 0

    def __get_num_sentences(self):
        """
        A function calculating number of available examples (sentences) in dataset file provided.

        :return: int, number of sentences available
        """
        return 1
        raise NotImplementedError

    def __examples(self):
        """
        A generator of examples iterating over the dataset with per sentence stepping.

        :yield: str, str: sentence, target translation
        """
        with open(self.dataset_path, "r") as input_file, open(self.targets_path, "r") as target_file:
            for input_line, target_line in zip(input_file, target_file):
                yield input_line, target_line

    def __load_next_inputs_maybe(self):
        """
        A function that loads new examples in the quantity equal to the requested batch size under the condition that
        previously issued questions have already been answered.
        """
        if self.__unanswered_questions_count == 0:
            sentences = list()
            self.translations = list()
            for _ in range(self.__batch_size):
                try:
                    sentence, translation = next(self.__example_iterator)
                except StopIteration:
                    raise utils.OutOfInstances("No more examples to process in the file provided.")
                sentences.append(sentence)
                self.translations.append(translation)
                self.__questions_count += 1
                self.__unanswered_questions_count += 1
            self.__current_inputs = sentences


    def reset(self):
        self.__example_iterator = self.__examples()
        self.__questions_count = 0
        self.__unanswered_questions_count = 0
        self.__current_inputs = None
        self.__exact_match_count = 0
        self.__f1_count = 0
        return True


    def get_input_array(self):
        self.__load_next_inputs_maybe()
        return self.__current_inputs


    def submit_prediction(self, id_in_batch: int, translation: string):
        """
        A function allowing for a submission of obtained results of NLP inference.

        :param id_in_batch: int, index in input batch that translation relates to
        :param translation: string, detokenized translation
        """
        target = self.translations[id_in_batch]
        metric = BLEU()
        print(metric.corpus_score(translation, [target]))
        
        self.__exact_match_count += 1 # TODO: Do the actual accuracy check
        self.__f1_count += 0
        self.__unanswered_questions_count -= 1

    def summarize_accuracy(self):
        """
        A function summarizing the accuracy achieved on the questions obtained with get_*_array() calls on which
        predicted answers were supplied with submit_predictions() function.

        :return: dict, dictionary containing a metric produced: bleu score
        """
        if self.__unanswered_questions_count != 0:
            utils.print_goodbye_message_and_die(
                "Answers for some of the issued questions have not been submitted.")

        exact_match = self.__exact_match_count / self.__questions_count
        print("\n Exact match = {:.3f}".format(exact_match))

        f1 = self.__f1_count / self.__questions_count
        print(" F1 = {:.3f}".format(f1))

        print(f"\nAccuracy figures above calculated on the basis of {self.__questions_count} translated sentences.")
        return {"exact_match": exact_match, "f1": f1} #TODO: BLEU, not this
        raise NotImplementedError
