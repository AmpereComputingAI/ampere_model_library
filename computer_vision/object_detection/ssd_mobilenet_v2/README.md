# SSD MobileNet v2

This folder contains the script to run SSD MobileNet v2 on COCO object detection task.

Variant supplied below for ONNX Runtime in two different precisions accepts input of shape 300x300.

Variant supplied below for ONNX Runtime in fp32 precision accepts input of shape 640x640.

The original paper on the architecture is available here: https://arxiv.org/abs/1801.04381

### Metrics

Based on 1000 images from COCO Dataset for TensorFlow framework in fp32 precision

| Metric                  | IoU       | Area   | maxDets |Score  |
|:---:                    |:---:      |:---:   |:---:    |:---:  |
| Average Precision  (AP) | 0.50:0.95 |    all | 100     | 0.208 |
| Average Precision  (AP) | 0.50      |    all | 100     | 0.307 |
| Average Precision  (AP) | 0.75      |    all | 100     | 0.220 |
| Average Precision  (AP) | 0.50:0.95 |  small | 100     | 0.017 |
| Average Precision  (AP) | 0.50:0.95 | medium | 100     | 0.136 |
| Average Precision  (AP) | 0.50:0.95 |  large | 100     | 0.478 |
| Average Recall     (AR) | 0.50:0.95 |    all |   1     | 0.189 |
| Average Recall     (AR) | 0.50:0.95 |    all |  10     | 0.233 |
| Average Recall     (AR) | 0.50:0.95 |    all | 100     | 0.234 |
| Average Recall     (AR) | 0.50:0.95 |  small | 100     | 0.020 |
| Average Recall     (AR) | 0.50:0.95 | medium | 100     | 0.152 |
| Average Recall     (AR) | 0.50:0.95 |  large | 100     | 0.531 |

Based on 1000 images from COCO Dataset for TensorFlow framework in int8 precision

| Metric                  | IoU       | Area   | maxDets |Score  |
|:---:                    |:---:      |:---:   |:---:    |:---:  |
| Average Precision  (AP) | 0.50:0.95 |    all | 100     | 0.194 |
| Average Precision  (AP) | 0.50      |    all | 100     | 0.306 |
| Average Precision  (AP) | 0.75      |    all | 100     | 0.209 |
| Average Precision  (AP) | 0.50:0.95 |  small | 100     | 0.017 |
| Average Precision  (AP) | 0.50:0.95 | medium | 100     | 0.132 |
| Average Precision  (AP) | 0.50:0.95 |  large | 100     | 0.441 |
| Average Recall     (AR) | 0.50:0.95 |    all |   1     | 0.184 |
| Average Recall     (AR) | 0.50:0.95 |    all |  10     | 0.243 |
| Average Recall     (AR) | 0.50:0.95 |    all | 100     | 0.243 |
| Average Recall     (AR) | 0.50:0.95 |  small | 100     | 0.023 |
| Average Recall     (AR) | 0.50:0.95 | medium | 100     | 0.174 |
| Average Recall     (AR) | 0.50:0.95 |  large | 100     | 0.551 |

Based on 1000 images from COCO Dataset for ONNX Runtime framework

| Metric                  | IoU       | Area   | maxDets |Score  |
|:---:                    |:---:      |:---:   |:---:    |:---:  |
| Average Precision  (AP) | 0.50:0.95 |    all | 100     | 0.264 |
| Average Precision  (AP) | 0.50      |    all | 100     | 0.390 |
| Average Precision  (AP) | 0.75      |    all | 100     | 0.284 |
| Average Precision  (AP) | 0.50:0.95 |  small | 100     | 0.025 |
| Average Precision  (AP) | 0.50:0.95 | medium | 100     | 0.184 |
| Average Precision  (AP) | 0.50:0.95 |  large | 100     | 0.619 |
| Average Recall     (AR) | 0.50:0.95 |    all |   1     | 0.228 |
| Average Recall     (AR) | 0.50:0.95 |    all |  10     | 0.290 |
| Average Recall     (AR) | 0.50:0.95 |    all | 100     | 0.292 |
| Average Recall     (AR) | 0.50:0.95 |  small | 100     | 0.030 |
| Average Recall     (AR) | 0.50:0.95 | medium | 100     | 0.206 |
| Average Recall     (AR) | 0.50:0.95 |  large | 100     | 0.670 |

### Dataset

Dataset can be downloaded from here: https://cocodataset.org/#download

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the model zoo directory,
as well as AIO_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/model_zoo
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
python3 run.py -m /path/to/model.pb -p fp32 --framework tf
```

Example command for ONNX Runtime: 

```
python3 run.py -m /path/to/model.onnx -p fp32 --framework ort
```