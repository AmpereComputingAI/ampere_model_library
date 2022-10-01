#
# -*- coding: utf-8 -*-
#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import sys
import os
import numpy as np
import argparse
import collections
import time
import math
import json
import datetime

from utils.tf import TFFrozenModelRunner
from utils.recommendation.widedeep import WideDeep

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.framework import ops
from tensorflow.core.framework import graph_pb2
from google.protobuf import text_format
from utils.misc import print_goodbye_message_and_die


def str2bool(v):
    if v.lower() in ('true'):
        return True
    else:
        return False


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument('--batch_size', type=int,
                        help='batch size for inference.Default is 512',
                        default=512,
                        dest='batch_size')
    parser.add_argument("-f", "--framework",
                        type=str, default="tf",
                        choices=["tf"],
                        help="specify the framework in which a model should be run")
    parser.add_argument('--num_intra_threads', type=int,
                        help='number of threads for an operator',
                        required=False,
                        default=28,
                        dest='num_intra_threads')
    parser.add_argument('--num_inter_threads', type=int,
                        help='number of threads across operators',
                        required=False,
                        default=2,
                        dest='num_inter_threads')
    parser.add_argument('--num_omp_threads', type=str,
                        help='number of threads to use',
                        required=False,
                        default=None,
                        dest='num_omp_threads')
    parser.add_argument("--dataset_path",
                        type=str, required=True,
                        help="path to a dataset")
    return parser.parse_args()


def run_tf_fp(model_path, batch_size, dataset_path):
    print(model_path)

    config = tf.compat.v1.ConfigProto(log_device_placement=False,
                                      inter_op_parallelism_threads=8,
                                      intra_op_parallelism_threads=1)
    graph = ops.Graph()
    graph_def = graph_pb2.GraphDef()

    filename, file_ext = os.path.splitext(model_path)

    batch_size = batch_size
    with open(model_path, "rb") as f:
        if file_ext == ".pbtxt":
            text_format.Merge(f.read(), graph_def)
        else:
            graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    numeric_feature_names = ["numeric_1"]
    string_feature_names = ["string_1"]

    full_features_names = numeric_feature_names + string_feature_names
    feature_datatypes = [tf.io.FixedLenSequenceFeature([], tf.float32, default_value=0.0, allow_missing=True)] + [
        tf.io.FixedLenSequenceFeature(
            [], tf.int64, default_value=0, allow_missing=True)]

    def input_fn(data_file, num_epochs, shuffle, batch_size):
        """Generate an input function for the Estimator."""

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
        dataset = tf.data.TFRecordDataset([data_file])
        if shuffle:
            dataset = dataset.shuffle(buffer_size=20000)
        dataset = dataset.batch(batch_size)
        dataset = dataset.map(_parse_function, num_parallel_calls=28)
        dataset = dataset.prefetch(batch_size * 10)
        return dataset

    data_file = dataset_path
    no_of_test_samples = sum(1 for _ in tf.compat.v1.python_io.tf_record_iterator(data_file))
    no_of_batches = math.ceil(float(no_of_test_samples) / batch_size)
    placeholder_list = ['import/new_numeric_placeholder:0', 'import/new_categorical_placeholder:0']
    input_tensor = [graph.get_tensor_by_name(name) for name in placeholder_list]
    output_name = "import/head/predictions/probabilities"
    output_tensor = graph.get_tensor_by_name("import/" + output_name + ":0")
    output_name1 = ["import/head/predictions/probabilities"]
    correctly_predicted = 0
    total_infer_consume = 0.0
    warm_iter = 100
    features_list = []
    print('here')

    runner = TFFrozenModelRunner(model_path, [output_name1])
    dataset = WideDeep(batch_size, dataset_path)

    with tf.compat.v1.Session(config=config, graph=runner.graph) as sess:
        res_dataset = input_fn(data_file, 1, False, batch_size)
        iterator = tf.compat.v1.data.make_one_shot_iterator(res_dataset)
        next_element = iterator.get_next()
        for i in range(int(no_of_batches)):
            batch = sess.run(next_element)
            features = batch[0:3]
            features_list.append(features)

    # print(features_list)
    print(type(features_list))
    print(len(features_list))
    print('here1')
    # quit()

    with tf.compat.v1.Session(config=config, graph=graph) as sess1:
        i = 0
        while True:
            if i >= no_of_batches:
                break
            if i > warm_iter:
                inference_start = time.time()
            logistic = sess1.run(output_tensor, dict(zip(input_tensor, features_list[i][0:2])))
            if i > warm_iter:
                infer_time = time.time() - inference_start
                total_infer_consume += infer_time

            i = i + 1
        inference_end = time.time()
    evaluate_duration = total_infer_consume
    latency = (1000 * batch_size * float(evaluate_duration) / float(no_of_test_samples - warm_iter * batch_size))
    throughput = (no_of_test_samples - warm_iter * batch_size) / evaluate_duration

    print('--------------------------------------------------')
    print('Total test records           : ', no_of_test_samples)
    print('Batch size is                : ', batch_size)
    print('Number of batches            : ', int(no_of_batches))
    print('Inference duration (seconds) : ', round(evaluate_duration, 4))
    print('Average Latency (ms/batch)   : ', round(latency, 4))
    print('Throughput is (records/sec)  : ', round(throughput, 3))
    print('--------------------------------------------------')


def run_tf_fp32(model_path, batch_size, dataset_path, **kwargs):
    return run_tf_fp(model_path, batch_size, dataset_path)


def main():
    args = parse_args()

    if args.framework == "tf":
        if args.model_path is None:
            print_goodbye_message_and_die(
                "a path to model is unspecified!")

        if args.precision == "fp32":
            run_tf_fp32(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
