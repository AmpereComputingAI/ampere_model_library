# RNNT

This folder contains the script to run RNNT on MLPerf speech-to-text task in PyTorch framework.

The original paper on the architecture is available here: https://arxiv.org/pdf/1811.06621.pdf


### Accuracy:

Based on 54 402 words from OpenSLR LibriSpeech Corpus for PyTorch framework

|   | &nbsp;&nbsp;&nbsp;&nbsp; Accuracy &nbsp;&nbsp;&nbsp;&nbsp;  |
|:---:|:---:|
| FP32  | 92.1 %  |


### Dataset and models

Dataset can be downloaded here: https://www.openslr.org/resources/12/dev-clean.tar.gz

PyTorch model in fp32 precision can be downloaded here: https://zenodo.org/record/3662521/files/DistributedDataParallel_1576581068.9962234-epoch-100.pt

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the Ampere Model Library directory,
as well as AIO_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/ampere_model_library
export AIO_NUM_THREADS=1
```

For the best experience we also recommend setting environment variables as specified below.

```
export LIBRISPEECH_DATASET_PATH=/path/to/dataset
```

Now you are able to run the run.py script. 

To get detailed information on the script's recognized arguments run it with -h flag for help.

For PyTorch implementation the precision (with a flag "-p") as well as framework (with a fla "--framework") have to be specified.\

Example command for PyTorch:

```
python3 run.py -p fp32 --framework pytorch
```
