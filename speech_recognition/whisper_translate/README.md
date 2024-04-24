# Whisper translate

This folder contains the script to run Whisper on speech-to-text translation task in PyTorch framework.

The original paper on the architecture is available here: https://arxiv.org/pdf/2212.04356.pdf


### Dataset

Download the Common Voice Corpus for the Japanese language here: https://commonvoice.mozilla.org/en/datasets

Extract the dataset:
```
tar -xvf ja.tar
```

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the Ampere Model Library directory,
as well as AIO_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/ampere_model_library
export AIO_NUM_THREADS=1
```

For the best experience we also recommend setting environment variables as specified below.

```
export COMMONVOICE_PATH=/path/to/dataset
```

Now you are able to run the run.py script. 

To get detailed information on the script's recognized arguments run it with -h flag for help.

For PyTorch implementation the size of the model (with a flag "-m") has to be specified.

Example command for PyTorch:

```
python3 run.py -m medium --timeout 600
```
