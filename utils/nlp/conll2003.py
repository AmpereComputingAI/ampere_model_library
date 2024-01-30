import numpy as np
import string

import utils.misc as utils
from utils.helpers import Dataset


class CoNLL2003(Dataset):
    """
    A class providing facilities for preprocessing and postprocessing of CoNLL-2003 dataset.
    """

    def __init__(self, batch_size: int, tokenize_func, detokenize_func, target_seq_size=None, dataset_path=None):

        if dataset_path is None:
            env_var = "CONLL2003_PATH"
            dataset_path = utils.get_env_variable(
                env_var, f"Path to CoNLL-2003 dataset has not been specified with {env_var} flag")

        self.__batch_size = batch_size
        self.__target_seq_size = target_seq_size
        self.__tokenize_func = tokenize_func
        self.__detokenize_func = detokenize_func
        self.__dataset = dataset_path
        self.__example_iterator = self.__examples()

        self.__texts_count = 0
        self.__unsubmitted_count = 0
        self.available_instances = sum(1 for _ in self.__examples())
        self.__current_inputs = None
        self.__ner_tags = None
        self.__f1_count = 0
        self.__exact_match_count = 0

    def __examples(self):
        with open(self.__dataset, encoding="utf-8") as f:
            tokens = []
            ner_tags = []
            for line in f:
                if line.startswith("-DOCSTART-") or line == "" or line == "\n":
                    if tokens:
                        yield tokens, ner_tags
                        tokens = []
                        ner_tags = []
                else:
                    # conll2003 tokens are space separated
                    splits = line.split(" ")
                    tokens.append(splits[0])
                    ner_tags.append(splits[3].rstrip())
            # last example
            yield tokens, ner_tags

    def __load_next_inputs_maybe(self):
        """
        A function that loads new examples in the quantity equal to the requested batch size under the condition that
        previously issued questions have already been answered.
        """
        if self.__unsubmitted_count == 0:
            texts = list()
            self.__ner_tags = list()
            examples_needed = self.__batch_size
            while examples_needed > 0:
                try:
                    tokens, ner_tags = next(self.__example_iterator)
                except StopIteration:
                    raise utils.OutOfInstances("No more examples to process in the CoNLL-2003 file provided.")

                examples_needed -= 1
                texts.append(tokens)
                self.__ner_tags.append(ner_tags)
                self.__texts_count += 1
                self.__unsubmitted_count += 1
            tokenized = self.__tokenize_func(texts)
            self.__current_inputs = tokenized

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
        self.__ner_tags = None
        self.__f1_count = 0
        self.__exact_match_count = 0
        return True

    def get_input_ids_array(self):
        return self.__get_input_array("input_ids")

    def get_attention_mask_array(self):
        return self.__get_input_array("attention_mask")

    def get_token_type_ids_array(self):
        return self.__get_input_array("token_type_ids")

    def get_offset_mapping_array(self):
        return self.__current_inputs["offset_mapping"]

    def submit_prediction(self, id_in_batch: int, prediction: string):
        """
        A function allowing for a submission of obtained results of NLP inference.

        :param id_in_batch: int, index in input batch that prediction relates to
        :param prediction: list of str, prediction
        """

        def f1_score(prediction, ground_truth):
            """
            A function calculating the F1 score betweed prediction and ground truth.

            :param prediction: str, answer (prediction)
            :param ground_truth: str, correct answer (gt)
            :return: float, f1 score
            """

            common = [1 for x, y in zip(prediction, ground_truth) if x == y]
            num_same = sum(common)
            if num_same == 0:
                return 0
            precision = 1.0 * num_same / len(prediction)
            recall = 1.0 * num_same / len(ground_truth)
            f1 = (2 * precision * recall) / (precision + recall)
            return f1

        def exact_match_score(prediction, ground_truth):
            """
            A function comparing lists of strings.

            :param prediction: list of str, answer (prediction)
            :param ground_truth: list of str, correct answer (gt)
            :return: bool, True if prediction and ground_truth are equal
            """
            return prediction == ground_truth

        def metric_max_over_ground_truths(metric_fn, prediction, ground_truth):
            """
            A function applying given metric function over provided ground_truths.

            :param metric_fn: function calculating a metric
            :param prediction: list of str with prediction
            :param ground_truth: list of strings, correct answer

            :return: float, max score obtained
            """
            return metric_fn(prediction, ground_truth)

        ground_truth = self.__ner_tags[id_in_batch]
        self.__exact_match_count += metric_max_over_ground_truths(exact_match_score, prediction, ground_truth)
        self.__f1_count += metric_max_over_ground_truths(f1_score, prediction, ground_truth)
        self.__unsubmitted_count -= 1

    def summarize_accuracy(self):
        """
        A function summarizing the accuracy achieved on the sequences obtained with get_*_array() calls on which
        predictions were supplied with submit_predictions() function.

        :return: dict, dictionary containing two metrics produced: exact_match signifying the ratio of perfect answers
        and f1 metric
        """
        if self.__unsubmitted_count != 0:
            utils.print_goodbye_message_and_die(
                "Predictions for some of the issued sequences have not been submitted.")

        exact_match = self.__exact_match_count / self.__texts_count
        #print("\n Exact match = {:.3f}".format(exact_match))

        f1 = self.__f1_count / self.__texts_count
        #print(" F1 = {:.3f}".format(f1))

        #print(f"\nAccuracy figures above calculated on the basis of {self.__texts_count} sequences predicted.")
        return {"exact_match": exact_match, "f1": f1}
