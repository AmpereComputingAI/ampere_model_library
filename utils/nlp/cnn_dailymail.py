# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import numpy as np
import re
import string
from collections import Counter
import pathlib
import utils.misc as utils
from utils.helpers import Dataset


class CNN_DailyMail(Dataset):
    """
    A class providing facilities for preprocessing and postprocessing of CNN/DailyMail dataset.
    """

    def __init__(self, batch_size: int, tokenize_func, detokenize_func, target_seq_size=None, dataset_path=None):

        if dataset_path is None:
            env_var = "CNN_DAILYMAIL_PATH"
            dataset_path = utils.get_env_variable(
                env_var, f"Path to CNN/DailyMail dataset has not been specified with {env_var} flag")

        self.__batch_size = batch_size
        self.__target_seq_size = target_seq_size
        self.__tokenize_func = tokenize_func
        self.__detokenize_func = detokenize_func
        self.__dataset = self.__load(dataset_path)
        self.__example_iterator = self.__examples()

        self.__texts_count = 0
        self.__unsubmitted_count = 0
        self.available_instances = sum(1 for _ in self.__load(dataset_path))
        self.__current_inputs = None
        self.__rouge2_count = 0

    def __load(self, dataset_path):
        """
        A function loading CNN/DailyMail dataset.

        :param dataset_path: str, path to directory containing the dataset
        :return: dictionary with nested dictionaries / lists, dataset
        """
        return pathlib.Path(dataset_path).rglob("*.story")

    def __examples(self):
        """
        A generator of examples iterating over the dataset with per story stepping.

        :yield: str, str: full story, summary
        """
        for story in self.__dataset:
            with open(story, "r") as f:
                text = ""
                summary = ""
                highlight = False
                for line in f:
                    if line.isspace():
                        continue
                    elif highlight:
                        summary += line
                        highlight = False
                    elif "@highlight" in line:
                        highlight = True
                        continue
                    else:
                        text += line

                yield text, summary

    def __load_next_inputs_maybe(self):
        """
        A function that loads new examples in the quantity equal to the requested batch size under the condition that
        previously issued questions have already been answered.
        """
        if self.__unsubmitted_count == 0:
            texts = list()
            self.__summaries = list()
            examples_needed = self.__batch_size
            while examples_needed > 0:
                try:
                    text, summary = next(self.__example_iterator)
                except StopIteration:
                    raise utils.OutOfInstances("No more examples to process in the CNN/DailyMail directory provided.")
                tokenized = self.__tokenize_func(text)
                if len(tokenized["input_ids"]) > 1024:
                    continue
                examples_needed -= 1
                texts.append(text)
                self.__summaries.append(summary)
                self.__texts_count += 1
                self.__unsubmitted_count += 1
            self.__current_inputs = self.__tokenize_func(texts)

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
        self.__texts_count = 0
        self.__unsubmitted_count = 0
        self.__current_inputs = None
        self.__rouge2_count = 0
        return True

    def get_input_ids_array(self):
        return self.__get_input_array("input_ids")

    def get_attention_mask_array(self):
        return self.__get_input_array("attention_mask")

    def get_token_type_ids_array(self):
        return self.__get_input_array("token_type_ids")

    def submit_prediction(self, id_in_batch: int, summary: string):
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

            def remove_tokens(text):
                return re.sub(r'<s>|</s>|<pad>', ' ', text)

            def white_space_fix(text):
                return ' '.join(text.split())

            def remove_punc(text):
                exclude = set(string.punctuation)
                return ''.join(ch for ch in text if ch not in exclude)

            def lower(text):
                return text.lower()

            return white_space_fix(remove_punc(remove_tokens(remove_articles(lower(answer_string)))))

        def rouge2_score(normalized_prediction, normalized_ground_truth):
            """
            A function calculating the ROUGE-2 score betweed normalized prediction and normalized ground truth.

            :param normalized_prediction: str, normalized answer (prediction)
            :param normalized_ground_truth: str, normalized correct answer (gt)
            :return: float, ROUGE-2 score
            """
            # prediction_tokens = normalized_prediction.split()
            # ground_truth_tokens = normalized_ground_truth.split()
            prediction_tokens = get_bigrams(normalized_prediction)
            ground_truth_tokens = get_bigrams(normalized_ground_truth)
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                return 0
            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            rouge2 = (2 * precision * recall) / (precision + recall)
            return rouge2

        def get_bigrams(text):
            """
            A function splitting text into bigrams (a sequence of two adjacent words).

            :param text: str, text to split
            :return: list, a list of bigrams
            """
            return [b for b in zip(text.split()[:-1], text.split()[1:])]

        ground_truth = self.__summaries[id_in_batch]
        self.__rouge2_count += rouge2_score(normalize(summary), normalize(ground_truth))
        self.__unsubmitted_count -= 1

    def _summarize_accuracy(self):
        """
        A function summarizing the accuracy achieved on the texts obtained with get_*_array() calls on which
        predicted summaries were supplied with submit_predictions() function.

        :return: dict, dictionary containing two metrics produced: exact_match signifying the ratio of perfect answers
        and ROUGE-2 metric
        """
        if self.__unsubmitted_count != 0:
            utils.print_goodbye_message_and_die(
                "Summaries for some of the issued texts have not been submitted.")

        rouge2 = self.__rouge2_count / self.__texts_count
        # print(" ROUGE-2 = {:.3f}".format(rouge2))

        # print(f"\nAccuracy figures above calculated on the basis of {self.__texts_count} summaries generated.")
        return {"rouge2": rouge2}
