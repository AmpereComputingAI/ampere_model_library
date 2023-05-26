# Retinanet

This folder contains the script to run Retinanet on COCO object detection task.

Variant supplied below for Pytorch in fp32 precision accepts input of shape 800x800.

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the model zoo directory,
as well as AIO_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/model_zoo
export AIO_NUM_THREADS=1
```

For the best experience we also recommend setting environment variables as specified below.

```
export OPENIMAGES_IMG_PATH=/path/to/images
export OPENIMAGES_ANNO_PATH=/path/to/annotations
```

Now you are able to run the run.py script.

To get detailed information on the script's recognized arguments run it with -h flag for help.

The path to model (with a flag "-m") as well as its precision (with a flag "-p") have to be specified.

Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.

Example command:

```
python3 run.py -m /path/to/model.onnx -p fp32 --framework pytorch
```