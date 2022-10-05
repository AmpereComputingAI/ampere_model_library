# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import math
import pickle
import collections

import numpy as np
import tensorflow as tf

import utils.misc as utils


class WideDeep:
    """
    A class providing facilities for preprocessing and postprocessing of ImageNet validation dataset.
    """

    def __init__(self, batch_size: int, dataset_path: str, config, runner):

        if dataset_path is None:
            env_var = "WIDEDEEP_PATH"
            dataset_path = utils.get_env_variable(
                env_var, f"Path to widedeep dataset has not been specified with {env_var} flag")

        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.available_instances = sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(self.dataset_path))
        self.no_of_batches = math.ceil(float(self.available_instances / self.batch_size))
        self.features_list = self.get_features_list(config, runner, self.no_of_batches)
        self.current_feature = 0
        self.correct = 0

        super().__init__()

    def input_fn(self, shuffle):
        """Generate an input function for the Estimator."""

        numeric_feature_names = ["numeric_1"]
        string_feature_names = ["string_1"]

        full_features_names = numeric_feature_names + string_feature_names + ["label"]
        feature_datatypes = [tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)] + \
                            [tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True)] + \
                            [tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True)]

        def _parse_function(proto):
            f = collections.OrderedDict(
                zip(full_features_names, feature_datatypes))
            parsed_features = tf.io.parse_example(proto, f)
            parsed_feature_vals_num = [tf.reshape(
                parsed_features["numeric_1"], shape=[-1, 13])]
            parsed_feature_vals_str = [tf.reshape(
                parsed_features["string_1"], shape=[-1, 2]) for i in string_feature_names]
            parsed_feature_vals = parsed_feature_vals_num + parsed_feature_vals_str

            # labels
            parsed_feature_vals_label = [tf.reshape(parsed_features[i], shape=[-1]) for i in ["label"]]
            parsed_feature_vals = parsed_feature_vals + parsed_feature_vals_label

            return parsed_feature_vals

        # Extract lines from input files using the Dataset API.
        dataset = tf.data.TFRecordDataset([self.dataset_path])
        if shuffle:
            dataset = dataset.shuffle(buffer_size=20000)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.map(_parse_function, num_parallel_calls=28)
        dataset = dataset.prefetch(self.batch_size * 10)
        return dataset

    def get_features_list(self, config, graph, no_of_batches):
        features_list = []
        with tf.compat.v1.Session(config=config, graph=graph) as sess:
            res_dataset = self.input_fn(False)
            iterator = tf.compat.v1.data.make_one_shot_iterator(res_dataset)
            next_element = iterator.get_next()
            for i in range(int(no_of_batches)):
                batch = sess.run(next_element)
                features_list.append(batch)

        file = open('important1', 'wb')
        pickle.dump(features_list, file)
        file.close()

        # open a file, where you stored the pickled data
        # file = open('important', 'rb')
        # dump information to that file
        # data = pickle.load(file)
        # close the file
        # file.close()
        #
        # print(features_list)
        # print(data)
        # print(type(data))
        # print(len(data))
        # print("-"*100)
        # print(type(features_list))
        # print(len(features_list))

        # if data == features_list:
        #     print('yuppie!')
        # else:
        #     print('buuuu')

        return features_list

    def get_input_array(self):

        try:
            input_array = self.features_list[self.current_feature][0:2]
        except IndexError:
            raise utils.OutOfInstances("no more features to process")
        return input_array

    def reset(self):
        self.current_feature = 0
        self.correct = 0
        return True

    def submit_predictions(self, output_array):
        predicted_labels = np.argmax(output_array['import/import/head/predictions/probabilities:0'], 1)
        self.correct = self.correct + np.sum(self.features_list[self.current_feature][2] == predicted_labels)
        self.current_feature += 1

    def summarize_accuracy(self):
        run_instances = self.current_feature * self.batch_size
        accuracy = float(self.correct) / float(run_instances)

        print("accuracy = {:.3f}".format(round((accuracy * 100), 4)))
        print("correct predictions = {:.3f}".format(self.correct))
        print("total predictions = {:.3f}".format(run_instances))
