# Copyright 2017-present Weichen Shen
# Copyright (c) 2022, Ampere Computing LLC

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

from utils.helpers import DatasetStub
from utils.misc import print_warning_message
from utils.recommendation.DeepCTR.deepctr.feature_column import SparseFeat, DenseFeat, get_feature_names


class CensusIncome(DatasetStub):
    def __init__(self, batch_size: int):
        self._batch_size = batch_size
        column_names = ['age', 'class_worker', 'det_ind_code', 'det_occ_code', 'education', 'wage_per_hour',
                        'hs_college', 'marital_stat', 'major_ind_code', 'major_occ_code', 'race', 'hisp_origin', 'sex',
                        'union_member', 'unemp_reason', 'full_or_part_emp', 'capital_gains', 'capital_losses',
                        'stock_dividends', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                        'det_hh_summ', 'instance_weight', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same',
                        'mig_prev_sunbelt', 'num_emp', 'fam_under_18', 'country_father', 'country_mother',
                        'country_self', 'citizenship', 'own_or_self', 'vet_question', 'vet_benefits', 'weeks_worked',
                        'year', 'income_50k']
        data = pd.read_csv(
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "DeepCTR/examples/census-income.sample"),
            header=None, names=column_names
        )
        data['label_income'] = data['income_50k'].map({' - 50000.': 0, ' 50000+.': 1})
        data['label_marital'] = data['marital_stat'].apply(lambda x: 1 if x == ' Never married' else 0)
        data.drop(labels=['income_50k', 'marital_stat'], axis=1, inplace=True)
        columns = data.columns.values.tolist()
        sparse_features = ['class_worker', 'det_ind_code', 'det_occ_code', 'education', 'hs_college', 'major_ind_code',
                           'major_occ_code', 'race', 'hisp_origin', 'sex', 'union_member', 'unemp_reason',
                           'full_or_part_emp', 'tax_filer_stat', 'region_prev_res', 'state_prev_res', 'det_hh_fam_stat',
                           'det_hh_summ', 'mig_chg_msa', 'mig_chg_reg', 'mig_move_reg', 'mig_same', 'mig_prev_sunbelt',
                           'fam_under_18', 'country_father', 'country_mother', 'country_self', 'citizenship',
                           'vet_question']
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

        self._dnn_feature_columns = fixlen_feature_columns
        linear_feature_columns = fixlen_feature_columns
        feature_names = get_feature_names(linear_feature_columns + self._dnn_feature_columns)

        instance_count = len(data[feature_names[0]])
        self._data = {}
        for name in feature_names:
            assert len(data[name]) == instance_count
            n, r = batch_size // instance_count, batch_size % instance_count
            array = data[name].to_numpy()
            self._data[name] = np.concatenate([array for _ in range(n)] + [array[:r]])

    def get_dnn_feature_columns(self):
        return self._dnn_feature_columns

    def get_inputs(self):
        return self._data

    def summarize_accuracy(self):
        print_warning_message("Accuracy testing unavailable for the Census Income dataset.")
