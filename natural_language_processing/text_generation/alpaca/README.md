# Alpaca

This folder contains the script to run Alpaca model on instruction following task in PyTorch framework.

### Dataset and model

Dataset can be found here: https://github.com/tatsu-lab/stanford_alpaca/blob/main/alpaca_data.json

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the ampere_model_library directory,
as well as AIO_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/ampere_model_library
export AIO_NUM_THREADS=1
```

For the best experience we also recommend setting environment variables as specified below.

```
export ALPACA_DATASET_PATH=/path/to/alpaca_data.json
```

Now you are able to run the run.py script.

To get detailed information on the script's recognized arguments run it with -h flag for help.

Framework (with a flag "--framework") has to be specified.
Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.

Example command for PyTorch:

```
TORCH_COMPILE=1 AIO_SKIP_MASTER_THREAD=1 python3 run.py -m path/to/alpaca_recovered
```
