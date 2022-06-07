# Hugging Face Summarization

This folder contains the script to run Hugging Face models on summarization task in PyTorch framework.

### Dataset and models

Dataset can be found here: https://cs.nyu.edu/~kcho/DMQA/
Download "Stories" files for both CNN and Daily Mail dataset.

Extract the dataset:
```
tar -xvf cnn_stories.tgz
tar -xvf dailymail_stories.tgz
```

Create a new directory and put both extracted folders there.

```
mkdir cnn_dailymail
mv cnn_stories dailymail_stories cnn_dailymail
```

PyTorch models can be found here: https://huggingface.co/models?library=pytorch&pipeline_tag=summarization&sort=downloads

Some of the supported models are:
```
sshleifer/distilbart-cnn-6-6 (default)
sshleifer/distilbart-cnn-12-6
facebook/bart-large-cnn
philschmid/bart-large-cnn-samsum
facebook/bart-large-xsum
sshleifer/distilbart-xsum-12-6
```

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the ampere_model_library directory,
as well as AIO_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/ampere_model_library
export AIO_NUM_THREADS=1
```

For the best experience we also recommend setting environment variables as specified below.

```
export CNN_DAILYMAIL_PATH=/path/to/dataset
```

Now you are able to run the run.py script.

To get detailed information on the script's recognized arguments run it with -h flag for help.

Framework (with a flag "--framework") has to be specified.
Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.

Example command for PyTorch:

```
python3 run.py -m sshleifer/distilbart-cnn-6-6 --framework pytorch
```
