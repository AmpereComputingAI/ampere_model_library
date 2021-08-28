from transformers import AutoTokenizer
import tensorflow as tf
import pandas as pd
import numpy as np


class MRPC:

    def __init__(self, model_name: str, batch_size: int, mrpc_dataset_path: None):

        self.__batch_size = batch_size
        self.__mrpc_dataset_path = mrpc_dataset_path
        self.__mrpc_dataset = pd.read_csv(self.__mrpc_dataset_path, sep=r'\t', engine='python').to_numpy()
        self.__tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.__current_sentence = 0
        self.available_instances = len(self.__mrpc_dataset)
        self.__label = None
        self.__correct = 0
        self.__incorrect = 0
        self.__count = 0
        self.latest_index = None

    def __get_sentence_index(self):

        self.__current_sentence += 1
        return self.__current_sentence

    def get_input_array(self):

        sequence_0 = [None] * self.__batch_size
        sequence_1 = [None] * self.__batch_size
        labels = [None] * self.__batch_size
        for i in range(self.__batch_size):
            self.latest_index = self.__get_sentence_index()
            sequence_0[i] = self.__mrpc_dataset[self.latest_index, 3]
            sequence_1[i] = self.__mrpc_dataset[self.latest_index, 4]
            labels[i] = int(self.__mrpc_dataset[self.latest_index, 0])

        input = self.__tokenizer(sequence_0, sequence_1, padding=True, truncation=True, return_tensors="tf")

        return input, labels

    def extract_prediction(self, output):
        predictions = np.argmax(output, axis=1)
        return predictions

    def submit_predictions(self, prediction, label):

        if label == prediction:
            self.__correct += 1
        else:
            self.__incorrect += 1
        self.__count += 1

    def summarize_accuracy(self):

        correct = self.__correct / self.__current_sentence
        print("\n Correct = {:.3f}".format(correct))

        print(f"\nAccuracy figures above calculated on the basis of {self.__current_sentence} pair of sentences.")
        return {"Correct": correct}

