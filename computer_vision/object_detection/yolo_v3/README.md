# YOLO v3

This folder contains the script to run YOLO v3 on COCO object detection task.

Variant supplied below for ONNX Runtime in fp32 precision accepts input of shape 416x416.

The original paper on the architecture is available here: https://arxiv.org/abs/1804.02767


### Metrics

Based on 1000 images from COCO Dataset

| Metric                  | IoU       | Area   | maxDets |Score  |
|:---:                    |:---:      |:---:   |:---:    |:---:  |
| Average Precision  (AP) | 0.50:0.95 |    all | 100     | 0.311 |
| Average Precision  (AP) | 0.50      |    all | 100     | 0.492 |
| Average Precision  (AP) | 0.75      |    all | 100     | 0.352 |
| Average Precision  (AP) | 0.50:0.95 |  small | 100     | 0.117 |
| Average Precision  (AP) | 0.50:0.95 | medium | 100     | 0.325 |
| Average Precision  (AP) | 0.50:0.95 |  large | 100     | 0.484 |
| Average Recall     (AR) | 0.50:0.95 |    all |   1     | 0.254 |
| Average Recall     (AR) | 0.50:0.95 |    all |  10     | 0.336 |
| Average Recall     (AR) | 0.50:0.95 |    all | 100     | 0.338 |
| Average Recall     (AR) | 0.50:0.95 |  small | 100     | 0.122 |
| Average Recall     (AR) | 0.50:0.95 | medium | 100     | 0.344 |
| Average Recall     (AR) | 0.50:0.95 |  large | 100     | 0.524 |

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

Example command: 

```
python3 run.py -m /path/to/model.onnx -p fp32 --framework ort
```