import numpy as np
import json
import re
import string
from collections import Counter
import pathlib
import utils.misc as utils


class OutOfInstances(Exception):
    """
    An exception class being raised as an error in case of lack of further images to process by the pipeline.
    """
    pass


class Squad_v1_1:
    """
    A class providing facilities for preprocessing and postprocessing of ImageNet validation dataset.
    """

    def __init__(self, batch_size: int, sequence_size: int, tokenize_func, detokenize_func, target_seq_size,
                 dataset_path=None, labels_path=None):

        if dataset_path is None:
            env_var = "SQUAD_V1.1_PATH"
            images_path = utils.get_env_variable(
                env_var, f"Path to Squad v1.1 .json file has not been specified with {env_var} flag")

        self.__batch_size = batch_size
        self.__seq_size = sequence_size
        self.__tokenize_func = tokenize_func
        self.__detokenize_func = detokenize_func
        self.__target_seq_size = target_seq_size
        self.__dataset = self.__verify_and_load(dataset_path)
        self.__example_iterator = self.__examples()

        self.__questions_count = 0
        self.__unanswered_questions_count = 0
        self.available_instances = self.__get_num_questions()
        self.__current_inputs = None
        self.__exact_match_count = 0
        self.__f1_count = 0

    def __get_num_questions(self):
        total_questions = 0
        for section in self.__dataset:
            for paragraph in section["paragraphs"]:
                total_questions += len(paragraph["qas"])
        return total_questions

    def __verify_and_load(self, dataset_path, expected_version="1.1"):
        """
        A function parsing validation file for ImageNet 2012 validation dataset.

        .txt file consists of 50000 lines each holding data on a single image: its file name and 1 label with class best
        describing image's content

        :param labels_path: str, path to file containing image file names and labels
        :param is1001classes: bool, parameter setting whether the tested model has 1001 classes (+ background) or
        original 1000 classes
        :return: list of strings, list of ints
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
        for section in self.__dataset:
            for paragraph in section["paragraphs"]:
                for qas in paragraph["qas"]:
                    print("$$$$$$$$$$$$$$$$$$$$4")
                    print(paragraph["context"], qas["question"], qas["answers"])
                    yield paragraph["context"], qas["question"], qas["answers"]

    def __load_next_inputs_maybe(self):
        if self.__unanswered_questions_count == 0:
            contextes = list()
            questions = list()
            self.__valid_answers = list()
            for _ in range(self.__batch_size):
                print("a")
                context, question, correct_answers = next(self.__example_iterator)
                print("b")
                contextes.append(context)
                questions.append(question)
                self.__questions_count += 1
                self.__unanswered_questions_count += 1
                self.__valid_answers.append(correct_answers)
            self.__current_inputs = self.__tokenize_func(questions, contextes)

    def __get_input_array(self, input_name):
        self.__load_next_inputs_maybe()

        input = self.__current_inputs[input_name]
        input_padded = np.empty([self.__batch_size, self.__target_seq_size])

        for i in range(self.__batch_size):
            space_to_pad = self.__target_seq_size - input[i].shape[0]
            input_padded[i] = np.pad(input[i], (0, space_to_pad), "constant", constant_values=0)

        return input_padded

    def get_input_ids_array(self):
        return self.__get_input_array("input_ids")

    def get_attention_mask_array(self):
        return self.__get_input_array("attention_mask")

    def get_token_type_ids_array(self):
        return self.__get_input_array("token_type_ids")

    def extract_answer(self, id_in_batch: int, answer_start_id, answer_end_id):
        answer = self.__current_inputs["input_ids"][id_in_batch][answer_start_id:answer_end_id+1]
        return self.__detokenize_func(answer)

    def submit_prediction(self, id_in_batch: int, answer):

        def normalize(answer_string):

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
            print(normalized_prediction)
            print(normalized_ground_truth)
            return normalized_prediction == normalized_ground_truth

        def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
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
        A function summarizing the accuracy achieved on the images obtained with get_input_array() calls on which
        predictions done where supplied with submit_predictions() function.
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
