# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import math
import time
import pickle
import bz2
import collections

import numpy as np
import tensorflow as tf

import utils.misc as utils


class WideDeep:
    """
    A class providing facilities for preprocessing and postprocessing of ImageNet validation dataset.
    """

    def __init__(self, batch_size: int, config=None, runner=None, dataset_path=None, tfrecords_path=None):

        if tfrecords_path is None:
            env_var = "WIDEDEEP_TFRECORDS_PATH"
            tfrecords_path = utils.get_env_variable(
                env_var, f"Path to tfrecords path has not been specified with {env_var} flag")

        self.batch_size = batch_size
        if self.batch_size in [1, 2, 4, 8, 16, 32, 50, 64, 100, 128, 200, 256]:
            if dataset_path is None:
                env_var = "WIDEDEEP_DATASET_PATH"
                dataset_path = utils.get_env_variable(
                    env_var, f"Path to widedeep dataset has not been specified with {env_var} flag")

        self.tfrecords_path = tfrecords_path
        self.dataset_path = dataset_path
        self.current_feature = 0
        self.correct = 0
        self.available_instances = sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(
            self.tfrecords_path))

        if self.batch_size in [1, 2, 4, 8, 16, 32, 50, 64, 100, 128, 200, 256]:
            self.features_list = self.unpickle()

        else:
            self.no_of_batches = math.ceil(float(self.available_instances / self.batch_size))
            self.features_list = self.get_features_list(config, runner, self.no_of_batches)

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
        dataset = tf.data.TFRecordDataset([self.tfrecords_path])
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

        return features_list

    def unpickle(self):
        file = bz2.BZ2File(self.dataset_path, 'rb')
        features_list = pickle.load(file)
        file.close()
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
