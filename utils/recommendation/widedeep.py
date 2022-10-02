# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import math
import collections

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
        self.no_of_test_samples = sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(self.dataset_path))
        self.no_of_batches = math.ceil(float(self.no_of_test_samples / self.batch_size))
        self.features_list = self.get_features_list(config, runner, self.no_of_batches)
        self.current_feature = 0
        super().__init__()

    def input_fn(self, shuffle):
        """Generate an input function for the Estimator."""

        numeric_feature_names = ["numeric_1"]
        string_feature_names = ["string_1"]

        full_features_names = numeric_feature_names + string_feature_names
        feature_datatypes = [tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)] + [
            tf.io.FixedLenSequenceFeature(
                [], tf.int64, default_value=0, allow_missing=True)]

        def _parse_function(proto):
            f = collections.OrderedDict(
                zip(full_features_names, feature_datatypes))
            parsed_features = tf.io.parse_example(proto, f)
            parsed_feature_vals_num = [tf.reshape(
                parsed_features["numeric_1"], shape=[-1, 13])]
            parsed_feature_vals_str = [tf.reshape(
                parsed_features["string_1"], shape=[-1, 2]) for i in string_feature_names]
            parsed_feature_vals = parsed_feature_vals_num + parsed_feature_vals_str
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
            # res_dataset = dataset.input_fn(False)
            res_dataset = self.input_fn(False)
            iterator = tf.compat.v1.data.make_one_shot_iterator(res_dataset)
            next_element = iterator.get_next()
            for i in range(int(no_of_batches)):
                batch = sess.run(next_element)
                features = batch[0:3]
                features_list.append(features)

        return features_list

    def get_input_array(self):
        input_array = self.features_list[self.current_feature][0:2]
        self.current_feature += 1
        return input_array

    def submit_predictions(self):
        pass

    def summarize_accuracy(self):
        pass
