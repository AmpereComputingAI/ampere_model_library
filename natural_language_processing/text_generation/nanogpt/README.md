# nanoGPT

This folder contains the script to run karpathy/nanoGPT model on text generation task in PyTorch framework.

### Dataset and model

Dataset can be found here: https://zenodo.org/record/2630551/files/lambada-dataset.tar.gz

Extract the dataset:
```
tar -xvf lambada-dataset.tgz
```

4 model sizes are supported: gpt2, gpt2-medium, gpt2-large, gpt2-xl.

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the ampere_model_library directory,
as well as AIO_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/ampere_model_library
export AIO_NUM_THREADS=1
```

For the best experience we also recommend setting environment variables as specified below.

```
export LAMBADA_PATH=/path/to/lambada_test_plain_text.txt
```

Now you are able to run the run.py script.

To get detailed information on the script's recognized arguments run it with -h flag for help.

Framework (with a flag "--framework") has to be specified.
Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.

Example command for PyTorch:

```
python3 run.py -m gpt2-medium  --framework pytorch
```
