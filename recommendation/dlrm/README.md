# DLRM


This folder contains the script to run DLRM on MLPerf recommendation task in PyTorch framework.

The original paper on the architecture is available here: https://arxiv.org/pdf/1906.00091.pdf

### Dataset and models

Dataset can be downloaded here: https://ailab.criteo.com/ressources/criteo-1tb-click-logs-dataset-for-mlperf/

Model can be downloaded here: https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt

Smaller debug model can be downloaded here: https://dlrm.s3-us-west-1.amazonaws.com/models/tb0875_10M.pt

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the model zoo directory,
as well as AIO_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/model_zoo
export AIO_NUM_THREADS=1
```

For the best experience we also recommend setting environment variables as specified below.

```
export CRITEO_DATASET_PATH=/path/to/dataset
```

Now you are able to run the run.py script.

To get detailed information on the script's recognized arguments run it with -h flag for help.

The precision (with a flag "-p") as well as framework (with a flag "--framework") have to be specified.
Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.

To run the smaller debug model, the flag --debug has to be specified.

Example command for PyTorch:

```
python3 run.py -m path/to/model.pt -p fp32 --framework pytorch
```
