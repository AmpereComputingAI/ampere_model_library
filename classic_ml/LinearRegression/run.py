# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import argparse
import numpy as np
from utils.benchmark import run_model
from classic_ml.tabular_dataset import TabularDataset
from utils.misc import print_goodbye_message_and_die, download_ampere_imagenet

def parse_args():
    parser = argparse.ArgumentParser(description="Run LinearRegression model.")
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str,
                        choices=["ort","sklearn"], required=True,
                        help="specify the framework in which a model should be run")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--data_path",
                        type=str,
                        help="path to csv file containing input data")
    parser.add_argument("--optimized",
                        type=bool,
                        default=False,
                        help="whether optimized LinearRegression is used or not")
    args = parser.parse_args()

    return args

def run_ort_fp(model_path, batch_size, num_runs, timeout, data_path):
    from utils.ort import OrtRunner

    def run_single_pass(ort_runner, dataset):
        X, y = next(dataset)
        
        ort_runner.set_input_tensor("input_tensor", X)
        y_hat = ort_runner.run()        
        dataset.submit_predictions(y, y_hat[0])

    dataset = TabularDataset(data_path, batch_size=batch_size, task='regression')
    runner = OrtRunner(model_path)
    
    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)

def run_sklearn_fp(model_path, batch_size, num_runs, timeout, data_path, optimized=False):
    global LinearRegression
    if optimized:
        print('Running in Optimized Mode')
        from sklearn.linear_model import  LinearRegression as LinearRegressionBase
        from sklearn.linear_model._base import safe_sparse_dot
        class LinearRegression(LinearRegressionBase):
            #overwriting decision func from original scikit learn and in this it skips checking and validation
            def _decision_function(self, X):
                return safe_sparse_dot(X, self.coef_.T, dense_output=True) + self.intercept_

    else:
        from sklearn.linear_model import  LinearRegression

    print(LinearRegression)
    from utils.sklearn import SklearnRunner

    def run_single_pass(sklearn_runner, dataset):
        X, y = next(dataset)
        y_hat = sklearn_runner.run(X)        
        dataset.submit_predictions(y, y_hat)

    dataset = TabularDataset(data_path, batch_size=batch_size, task='regression')
    runner = SklearnRunner(model_path)
    
    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_ort_fp32(model_path, batch_size, num_runs, timeout, data_path, **kwargs):
    return run_ort_fp(model_path, batch_size, num_runs, timeout, data_path)

def run_sklearn_fp32(model_path, batch_size, num_runs, timeout, data_path, **kwargs):
    return run_sklearn_fp(model_path, batch_size, num_runs, timeout, data_path, **kwargs)



def main():
    args = parse_args()
    
    if args.framework == "ort":
        if args.precision == "fp32":
            run_ort_fp32(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)
    elif args.framework == "sklearn":
        if args.precision == "fp32":
            run_sklearn_fp32(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)
    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
