# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import numpy as np
import pathlib
import os
import math
import collections

import tensorflow as tf
import utils.misc as utils
import utils.cv.pre_processing as pp
from utils.cv.dataset import ImageDataset

dataset_path = '/ampere/Downloads/widedeep/large_kaggle_display_advertising_challenge_dataset/eval_processed_data.tfrecords'
class WideDeep(ImageDataset):
    """
    A class providing facilities for preprocessing and postprocessing of ImageNet validation dataset.
    """

    def __init__(self, batch_size: int, dataset_path: str):

        if dataset_path is None:
            env_var = "WIDEDEEP_PATH"
            dataset_path = utils.get_env_variable(
                env_var, f"Path to widedeep dataset has not been specified with {env_var} flag")

        self.batch_size = batch_size
        self.dataset_path = dataset_path
        self.train_file = os.path.join('/Users/marcel/Downloads/widedeep_data', 'adult.data')
        self.test_file = os.path.join('/Users/marcel/Downloads/widedeep_data', 'adult.test')
        # self.numeric_feature_names = ["numeric_1"]
        # self.string_feature_names = ["string_1"]
        # self.full_features_names = self.numeric_feature_names + self.string_feature_names
        # self.feature_datatypes = [tf.io.FixedLenSequenceFeature([],
        #                                                         tf.float32, default_value=0.0, allow_missing=True)] + \
        #                          [tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True)]
        # self.no_of_test_samples = sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(self.dataset_path))
        # self.no_of_batches = math.ceil(float(self.no_of_test_samples) / self.batch_size)
        super().__init__()

    numeric_feature_names = ["numeric_1"]
    string_feature_names = ["string_1"]
    full_features_names = numeric_feature_names + string_feature_names
    feature_datatypes = [tf.io.FixedLenSequenceFeature([],
                                                            tf.float32, default_value=0.0, allow_missing=True)] + \
                             [tf.io.FixedLenSequenceFeature([], tf.int64, default_value=0, allow_missing=True)]
    no_of_test_samples = sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(dataset_path))
    no_of_batches = math.ceil(float(no_of_test_samples) / 1)

    def input_fn(self, shuffle, batch_size):
        """Generate an input function for the Estimator."""

        def _parse_function(proto):
            f = collections.OrderedDict(
                zip(self.full_features_names, self.feature_datatypes))
            parsed_features = tf.io.parse_example(proto, f)
            parsed_feature_vals_num = [tf.reshape(
                parsed_features["numeric_1"], shape=[-1, 13])]
            parsed_feature_vals_str = [tf.reshape(
                parsed_features["string_1"], shape=[-1, 2]) for i in self.string_feature_names]
            parsed_feature_vals = parsed_feature_vals_num + parsed_feature_vals_str
            return parsed_feature_vals

        # Extract lines from input files using the Dataset API.
        dataset = tf.data.TFRecordDataset([self.dataset_path])
        if shuffle:
            dataset = dataset.shuffle(buffer_size=20000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(_parse_function, num_parallel_calls=28)
        dataset = dataset.prefetch(batch_size * 10)
        return dataset
