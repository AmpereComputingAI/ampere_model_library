# Classical ML Models

This folder has two versions of running the sklearn model. Profiling the sklearn's linear regression model seems to spend most of the time in checking and validation(check_predict,asserts, check_arrays etc) and getting rid of this can reduce the time by half. To do so, we can overwrite decision function from sklearn to avoid checking and validation.

Each model has been trained on specific dataset and it should be tested using only that specific dataset. For ex Regression models are using diabetes dataset and classification models are using iris dataset.

We have two versions of the model, one is the regular sklearn model and the other is optimized version of sklearn model meaning it has been saved by overwriting the decision function. Use --optimized flag to select the model. With Optimized -True we can use linear_regression_optmized.joblib and with --optmized False we can use linear_regression_sklearn.py.Models can be downloaded from here:

```
https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/sklearn2onnx/linear_regression_optimized.joblib
https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/sklearn2onnx/linear_regression_sklearn.joblib
```

Dataset can be downloaded from here:
```
https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/sklearn2onnx/diabetes_test.csv
https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/sklearn2onnx/iris_test.csv

```

### Metrics

Linear Regression with sklearn

```
                              mean           median       90th-percentile
  Latency           [ms]:       0.03  /        0.04  /        0.07
 Throughput [samples/s]:   19170.11  /    22429.43  /    13400.33
 ```

Linear Regression with optimized sklearn
```
                               mean  /      median  / 90th-percentile
 Latency           [ms]:       0.01  /        0.01  /        0.01
 Throughput [samples/s]:  144860.55  /   167772.16  /   155344.59

 ```

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the model zoo directory,
as well as OMP_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/ampere_model_library
export OMP_NUM_THREADS=<numofcpus>

```

Now you are able to run the run.py script. 

To get detailed information on the script's recognized arguments run it with -h flag for help.

The path to model (with a flag "-m") as well as its precision (with a flag "-p") have to be specified.

Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.

Example command for sklearn: 



```
python run.py -m /path/to/model.joblib -p fp32 --optimized True/False --framework sklearn --data_path /path/to/*.csv

```
