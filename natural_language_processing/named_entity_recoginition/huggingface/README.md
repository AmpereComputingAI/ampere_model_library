# Hugging Face Named Entity Recognition

This folder contains the script to run Hugging Face models on named entity recognition task in PyTorch framework.

### Dataset and models

Dataset can be downloaded from here: https://data.deepai.org/conll2003.zip

```
wget https://data.deepai.org/conll2003.zip
```

Extract the dataset:
```
unzip conll2003.zip
```

PyTorch models can be found here: https://huggingface.co/models?library=pytorch&pipeline_tag=token-classification&sort=downloads

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the ampere_model_library directory,
as well as AIO_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/ampere_model_library
export AIO_NUM_THREADS=1
```

For the best experience we also recommend setting environment variables as specified below.

```
export CONLL2003_PATH=/path/to/dataset
```

Now you are able to run the run.py script.

To get detailed information on the script's recognized arguments run it with -h flag for help.

Framework (with a flag "--framework") has to be specified.
Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.

Example command for PyTorch:

```
python3 run.py -m dslim/bert-large-NER --framework pytorch
```
