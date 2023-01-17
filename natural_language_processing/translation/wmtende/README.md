This folder contains the script to run CTranslate2 models on WMT en-de translation task.

### Accuracy:

based on 1000 sentences

|   | &nbsp;&nbsp;&nbsp;&nbsp; BLEU&nbsp;&nbsp;&nbsp;&nbsp;  |
|:---:|:---:|
| FP32  | 29.025  |
| FP16  | 29.025  |
| INT8  | 29.017  |


### Dataset

Dataset can be downloaded here: https://www.statmt.org/wmt14/training-parallel-nc-v9.tgz


### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the model zoo directory,
as well as AIO_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/model_zoo
export AIO_NUM_THREADS=1
```

For the best experience we also recommend setting environment variables as specified below.

```
export WMT_DATASET_PATH=/path/to/dataset
export WMT_TARGETS_PATH=/path/to/targets
```

Now you are able to run the run.py script. 

To get detailed information on the script's recognized arguments run it with -h flag for help.

The precision (with a flag "-p") as well as framework (with a flag "--framework") and path to the tokenizers (with a flag "--tokenizer_path") have to be specified.
Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.


Example command: 

```
python3 run.py -p fp32 --framework ctranslate --tokenizer_path /path/to/tokenizer
```