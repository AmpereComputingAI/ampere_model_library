# YOLO v4

This folder contains the script to run YOLO v4 on COCO object detection task.

Variants supplied below for TensorFlow and ONNXRuntime accept input of shape 416x416.

The original paper on the architecture is available here: https://arxiv.org/pdf/2004.10934v1.pdf

### Metrics

Based on 1000 images from COCO Dataset for TensorFlow framework in fp32 precision

| Metric                  | IoU       | Area   | maxDets |Score  |
|:---:                    |:---:      |:---:   |:---:    |:---:  |
| Average Precision  (AP) | 0.50:0.95 |    all | 100     | 0.454 |
| Average Precision  (AP) | 0.50      |    all | 100     | 0.644 |
| Average Precision  (AP) | 0.75      |    all | 100     | 0.509 |
| Average Precision  (AP) | 0.50:0.95 |  small | 100     | 0.208 |
| Average Precision  (AP) | 0.50:0.95 | medium | 100     | 0.499 |
| Average Precision  (AP) | 0.50:0.95 |  large | 100     | 0.647 |
| Average Recall     (AR) | 0.50:0.95 |    all |   1     | 0.334 |
| Average Recall     (AR) | 0.50:0.95 |    all |  10     | 0.490 |
| Average Recall     (AR) | 0.50:0.95 |    all | 100     | 0.499 |
| Average Recall     (AR) | 0.50:0.95 |  small | 100     | 0.227 |
| Average Recall     (AR) | 0.50:0.95 | medium | 100     | 0.543 |
| Average Recall     (AR) | 0.50:0.95 |  large | 100     | 0.708 |

Based on 1000 images from COCO Dataset for ONNXRuntime framework in fp32 precision

| Metric                  | IoU       | Area   | maxDets |Score  |
|:---:                    |:---:      |:---:   |:---:    |:---:  |
| Average Precision  (AP) | 0.50:0.95 |    all | 100     | 0.454 |
| Average Precision  (AP) | 0.50      |    all | 100     | 0.644 |
| Average Precision  (AP) | 0.75      |    all | 100     | 0.509 |
| Average Precision  (AP) | 0.50:0.95 |  small | 100     | 0.208 |
| Average Precision  (AP) | 0.50:0.95 | medium | 100     | 0.499 |
| Average Precision  (AP) | 0.50:0.95 |  large | 100     | 0.647 |
| Average Recall     (AR) | 0.50:0.95 |    all |   1     | 0.334 |
| Average Recall     (AR) | 0.50:0.95 |    all |  10     | 0.490 |
| Average Recall     (AR) | 0.50:0.95 |    all | 100     | 0.499 |
| Average Recall     (AR) | 0.50:0.95 |  small | 100     | 0.227 |
| Average Recall     (AR) | 0.50:0.95 | medium | 100     | 0.543 |
| Average Recall     (AR) | 0.50:0.95 |  large | 100     | 0.708 |

### Dataset and model

Dataset can be downloaded from here: https://cocodataset.org/#download

#### Tensorflow

Convert darknet weights using this repo:
https://github.com/hunglc007/tensorflow-yolov4-tflite

Download link in the repo is broken, here's a working one:
`wget https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights`

#### ONNXRuntime

First get TF model, then convert using this tutorial:
https://learn.microsoft.com/cs-cz/windows/ai/windows-ml/tutorials/tensorflow-convert-model

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the Ampere Model Library directory,
as well as AIO_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/ampere_model_library
export AIO_NUM_THREADS=1
```

For the best experience we also recommend setting environment variables as specified below.

```
export COCO_IMG_PATH=/path/to/images
export COCO_ANNO_PATH=/path/to/annotations
```

Now you are able to run the run.py script. 

To get detailed information on the script's recognized arguments run it with -h flag for help.

The path to model (with a flag "-m") as well as its precision (with a flag "-p") have to be specified.

Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.

Example command for TensorFlow: 

```
python3 run.py -m /path/to/model -p fp32 --framework tf
```

Example command for ONNXRuntime: 

```
python3 run.py -m /path/to/model -p fp32 --framework ort
```