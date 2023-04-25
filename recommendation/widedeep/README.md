# Wide Deep

you can acquire the model from here:
```
wget https://storage.googleapis.com/intel-optimized-tensorflow/models/v1_8/wide_deep_fp32_pretrained_model.pb
```

### Dataset

The Large Kaggle Display Advertising Challenge Dataset will be used for inference. The data is from Criteo and has a field indicating if an ad was clicked (1) or not (0), along with integer and categorical features.

Download the Large Kaggle Display Advertising Challenge Dataset from Criteo Labs or the links provided below

```
wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/eval.csv
wget https://storage.googleapis.com/dataset-uploader/criteo-kaggle/large_version/train.csv
```

now run the script for data preprocessing

```
wget https://raw.githubusercontent.com/IntelAI/models/master/datasets/large_kaggle_advertising_challenge/preprocess_csv_tfrecords.py

python preprocess_csv_tfrecords.py --inputcsv-datafile eval.csv --calibrationcsv-datafile train.csv --outputfile-name preprocessed_eval
```


### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the model zoo directory,
as well as DLS_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/model_zoo
export DLS_NUM_THREADS=1
```

For the best experience we also recommend setting environment variables as specified below.

```
export WIDEDEEP_TFRECORDS_PATH=/path/to/preprocessed_data.tfrecords
```

Now you are able to run the run.py script. 

To get detailed information on the script's recognized arguments run it with -h flag for help.

The precision (with a flag "-p") as well as framework (with a flag "--framework") have to be specified.\
Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.


Example command: 

```
python3 run.py -m /path/to/model -p fp32 --framework tf
```