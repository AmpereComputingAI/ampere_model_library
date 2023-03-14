# Classical ML Models

This folder contains the script to run Traditional ML models from scikit-learn using onnx framework.

Each model has been trained on specific dataset and it should be tested using only that specific dataset. For ex Regression models are using diabetes dataset and classification models are using iris dataset.

Models can be downloaded from here:

```
https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/sklearn2onnx/DecisionTreeRegressor_model.onnx
https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/sklearn2onnx/linear_regression_model.onnx
https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/sklearn2onnx/logistic_regression_model.onnx
https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/sklearn2onnx/DecisionTreeClassifier_model.onnx
```

Dataset can be downloaded from here:
```
https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/sklearn2onnx/diabetes_test.csv
https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/sklearn2onnx/iris_test.csv

```

### Metrics

Linear Regression

 ```                             mean  /      median  / 90th-percentile
 Latency           [ms]:       0.01  /        0.01  /        0.01
 Throughput [samples/s]:   80442.76  /    91180.52  /    82241.25
 ```

Logistic Regression

 ```                              mean  /      median  / 90th-percentile
 Latency           [ms]:       0.02  /        0.02  /        0.02
 Throughput [samples/s]:   56281.19  /    61680.94  /    55924.05
 ```

DecisionTreeRegressor

 ```                            mean  /      median  / 90th-percentile
 Latency           [ms]:       0.01  /        0.01  /        0.01
 Throughput [samples/s]:   82953.00  /    91180.52  /    82241.25
 ```
DecisionTreeClassifier

 ```                             mean  /      median  / 90th-percentile
 Latency           [ms]:       0.02  /        0.01  /        0.02
 Throughput [samples/s]:   58106.38  /    67650.06  /    61680.94
 ```

### model

Model description can be found here : https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html for all the models.

Models can be downloaded from here:
```
 https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/sklearn2onnx/

```
Link for particular model is for ex: 
```
https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/sklearn2onnx/DecisionTreeRegressor_model.onnx

```

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the model zoo directory,
as well as AIO_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/ampere_model_library
export AIO_NUM_THREADS=<numofcpus>
export OMP_NUM_THREADS=<numofcpus>
```

Now you are able to run the run.py script. 

To get detailed information on the script's recognized arguments run it with -h flag for help.

The path to model (with a flag "-m") as well as its precision (with a flag "-p") have to be specified.

Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.

Example command for TensorFlow: 

```
python run.py -m /path/to/model.onnx -p fp32 --framework ort --data_path /path/to/*.csv
```
