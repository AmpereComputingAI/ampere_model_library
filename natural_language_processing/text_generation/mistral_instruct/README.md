# Mistral Instruct

This folder contains the script to run Mistral 7B Instruct v0.2 model on instruction following task in PyTorch framework.

### Dataset

Dataset can be found here: https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/main/alpaca_data.json

### Model

The original paper on the architecture is available here: https://arxiv.org/abs/2310.06825

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the ampere_model_library directory,
as well as AIO_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/ampere_model_library
export AIO_NUM_THREADS=64
```

For the best experience we also recommend setting environment variables as specified below.

```
export ALPACA_DATASET_PATH=/path/to/alpaca_data.json
```

Now you are able to run the run.py script.

To get detailed information on the script's recognized arguments run it with -h flag for help.

Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.

Example command for PyTorch:

```
python3 run.py
```
