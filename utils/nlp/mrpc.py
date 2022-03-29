import numpy as np
import pandas as pd
import tensorflow as tf
from transformers import AutoTokenizer

import utils.misc as utils


class MRPC:
    """
    A class providing facilities for preprocessing and postprocessing of MRPC test dataset.
    """

    def __init__(self, model_name: str, batch_size: int, mrpc_dataset_path: None):

        if mrpc_dataset_path is None:
            env_var = "MRPC_PATH"
            mrpc_dataset_path = utils.get_env_variable(
                env_var, f"Path to MRPC dataset directory has not been specified with {env_var} flag")

        self.__batch_size = batch_size
        self.__mrpc_dataset_path = mrpc_dataset_path
        self.__mrpc_dataset = pd.read_csv(self.__mrpc_dataset_path, sep=r'\t', engine='python').to_numpy()
        self.__tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.__current_sentence = 0
        # self.available_instances = len(self.__mrpc_dataset)
        self.__correct = 0
        self.latest_index = None

    def __get_sentence_index(self):
        """
        A function providing index to the pair of sentences in mrpc dataset.

        :return: int, index of the current pair of sentences
        """

        self.__current_sentence += 1
        return self.__current_sentence

    def get_input_array(self):
        """
        A function returning a tokenized input for the NLP model.

        :return: transformers.tokenization_utils_base.BatchEncoding, an object containing tokenized inputs
        :return: list, list containing labels to the sentences
        initialization
        """

        sequence_0 = [None] * self.__batch_size
        sequence_1 = [None] * self.__batch_size
        labels = [None] * self.__batch_size

        try:
            for i in range(self.__batch_size):
                self.latest_index = self.__get_sentence_index()
                sequence_0[i] = self.__mrpc_dataset[self.latest_index, 3]
                sequence_1[i] = self.__mrpc_dataset[self.latest_index, 4]
                labels[i] = int(self.__mrpc_dataset[self.latest_index, 0])
        except IndexError:
            raise utils.OutOfInstances("No more sentences in the MRPC dataset to be processed")

        input = self.__tokenizer(sequence_0, sequence_1, padding=True, truncation=True, return_tensors="tf")

        return input, labels

    def extract_prediction(self, output):
        """
        A function extracting the predictions of the NLP model

        :return: a NumPy array containing the predictions (a 1 or 0 value)
        """
        predictions = tf.argmax(output[0], axis=1).numpy()
        return predictions

    def submit_predictions(self, prediction, label):
        """
       A function meant for submitting predictions for a given pair of sentences.

       :param prediction: int, a prediction number (0 or 1)
       :param label: int, a prediction number (0 or 1)
       """
        if label == prediction:
            self.__correct += 1

    def summarize_accuracy(self):
        """
        A function summarizing the obtained accuracy for the model

        A function summarizing the accuracy achieved on the pair of sentences obtained with get_input_array() calls on
        which predictions done where supplied with submit_predictions() function.
        :return: dict, a dictionary containing the accuracy
        """

        correct = self.__correct / self.__current_sentence
        print("\n Accuracy = {:.3f}".format(correct))

        print(f"\nAccuracy figures above calculated on the basis of {self.__current_sentence} pair of sentences.")
        return {"Correct": correct}

