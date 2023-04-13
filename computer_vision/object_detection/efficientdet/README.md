# EfficientDet

This folder contains the script to run EfficientDet on COCO object detection task.

Variant supplied below for TFLite accepts input of shape 448x448.

The original paper on the architecture is available here: https://arxiv.org/pdf/1911.09070.pdf

### Metrics

Based on 1000 images from COCO Dataset for TensorFlow framework in int8 precision

| Metric                  | IoU       | Area   | maxDets |Score  |
|:---:                    |:---:      |:---:   |:---:    |:---:  |
| Average Precision  (AP) | 0.50:0.95 |    all | 100     | 0.432 |
| Average Precision  (AP) | 0.50      |    all | 100     | 0.624 |
| Average Precision  (AP) | 0.75      |    all | 100     | 0.467 |
| Average Precision  (AP) | 0.50:0.95 |  small | 100     | 0.174 |
| Average Precision  (AP) | 0.50:0.95 | medium | 100     | 0.477 |
| Average Precision  (AP) | 0.50:0.95 |  large | 100     | 0.678 |
| Average Recall     (AR) | 0.50:0.95 |    all |   1     | 0.330 |
| Average Recall     (AR) | 0.50:0.95 |    all |  10     | 0.482 |
| Average Recall     (AR) | 0.50:0.95 |    all | 100     | 0.494 |
| Average Recall     (AR) | 0.50:0.95 |  small | 100     | 0.209 |
| Average Recall     (AR) | 0.50:0.95 | medium | 100     | 0.548 |
| Average Recall     (AR) | 0.50:0.95 |  large | 100     | 0.743 |

### Dataset and model

Dataset can be downloaded from here: https://cocodataset.org/#download

TFLite model in int8 precision can be downloaded here: https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/default/1

```
wget -O lite-model_efficientdet_lite2_detection_default_1.tflite https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/default/1?lite-format=tflite
```

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the model zoo directory,
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
python3 run.py -m /path/to/model.pb -p int8 --framework tf
```
