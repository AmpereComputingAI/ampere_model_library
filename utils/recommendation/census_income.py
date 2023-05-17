# Copyright 2017-present Weichen Shen
# Copyright (c) 2022, Ampere Computing LLC

import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from utils.helpers import DatasetStub
from utils.misc import print_warning_message, OutOfInstances
from utils.recommendation.DeepCTR.deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names


class CensusIncome(DatasetStub):
    def __init__(self, batch_size: int, dataset_path):
        self._batch_size = batch_size
        data = pd.read_csv(dataset_path)

        data['label_income'] = data['income'].map({'<=50K': 0, '>50K': 1})
        data['label_marital'] = data['marital.status'].apply(lambda x: 1 if x == 'Never-married' else 0)
        data.drop(labels=['income', 'marital.status'], axis=1, inplace=True)

        columns = data.columns.values.tolist()
        sparse_features = [
            'workclass', 'fnlwgt', 'education', 'occupation', 'relationship', 'race', 'sex', 'native.country'
        ]
        dense_features = [col for col in columns if
                          col not in sparse_features and col not in ['label_income', 'label_marital']]

        data[sparse_features] = data[sparse_features].fillna('-1', )
        data[dense_features] = data[dense_features].fillna(0, )
        mms = MinMaxScaler(feature_range=(0, 1))
        data[dense_features] = mms.fit_transform(data[dense_features])

        for feat in sparse_features:
            lbe = LabelEncoder()
            data[feat] = lbe.fit_transform(data[feat])

        fixlen_feature_columns = [SparseFeat(feat, data[feat].max() + 1, embedding_dim=4) for feat in sparse_features] \
                                 + [DenseFeat(feat, 1, ) for feat in dense_features]

        self.dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns

        feature_names = get_feature_names(linear_feature_columns + self.dnn_feature_columns)

        self.train_set, self.test_set = train_test_split(data, test_size=0.2, random_state=2020)
        self.train_model_input = {name: self.train_set[name] for name in feature_names}
        self.test_model_input = {name: self.test_set[name] for name in feature_names}

        self.available_instances = len(self.test_model_input[feature_names[0]])
        self._current_instance = 0
        self._results = {}

    def get_inputs(self):
        if self._current_instance + self._batch_size > self.available_instances:
            raise OutOfInstances
        inputs = {}
        for key, val in self.test_model_input.items():
            if val.dtype == "int64":
                inputs[key] = np.expand_dims(
                    val[self._current_instance:self._current_instance+self._batch_size].astype("int32"), axis=1)
            elif val.dtype == "float64":
                inputs[key] = np.expand_dims(
                    val[self._current_instance:self._current_instance+self._batch_size].astype("float32"), axis=1)
            else:
                raise TypeError(f"unexpected dtype -> {val.dtype}")
        self._current_instance += self._batch_size
        return inputs

    def submit_results(self, results):
        for label in ["label_marital", "label_income"]:
            if label in self._results.keys():
                self._results[label] = np.concatenate((self._results[label], results[label]))
            else:
                self._results[label] = results[label]

    def reset(self):
        self._current_instance = 0
        self._results = {}
        return True

    def summarize_accuracy(self):
        summary = ""
        metrics = {}
        for label, metric in [("label_marital", "marital AUC"), ("label_income", "income AUC")]:
            metrics[metric] = roc_auc_score(self.test_set[label][:self._current_instance], self._results[label])
            summary += "\n {:<11} = {:.3f}".format(metric, metrics[metric])
        print(summary)
        print(f"\nAccuracy figures above calculated on the basis of {self._current_instance} samples.")
        return metrics
