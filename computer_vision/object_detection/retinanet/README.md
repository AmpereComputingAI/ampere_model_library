# Retinanet

This folder contains the script to run Retinanet on OpenImages object detection task.

Variant supplied below for Pytorch in fp32 precision accepts input of shape 800x800.

The original paper on the architecture is available here: https://arxiv.org/pdf/1708.02002.pdf


### Metrics

Based on 320 images from OpenImages Dataset

| Metric                  | IoU       | Area   | maxDets |Score  |
|:---:                    |:---:      |:---:   |:---:    |:---:  |
| Average Precision  (AP) | 0.50:0.95 |    all | 100     | 0.523 |
| Average Precision  (AP) | 0.50      |    all | 100     | 0.694 |
| Average Precision  (AP) | 0.75      |    all | 100     | 0.564 |
| Average Precision  (AP) | 0.50:0.95 |  small | 100     | 0.080 |
| Average Precision  (AP) | 0.50:0.95 | medium | 100     | 0.214 |
| Average Precision  (AP) | 0.50:0.95 |  large | 100     | 0.565 |
| Average Recall     (AR) | 0.50:0.95 |    all |   1     | 0.490 |
| Average Recall     (AR) | 0.50:0.95 |    all |  10     | 0.640 |
| Average Recall     (AR) | 0.50:0.95 |    all | 100     | 0.654 |
| Average Recall     (AR) | 0.50:0.95 |  small | 100     | 0.188 |
| Average Recall     (AR) | 0.50:0.95 | medium | 100     | 0.344 |
| Average Recall     (AR) | 0.50:0.95 |  large | 100     | 0.695 |

### Dataset and model

Dataset can be downloaded using this script: https://github.com/mlcommons/inference/blob/r2.1/vision/classification_and_detection/tools/openimages_mlperf.sh

Pytorch model in fp32 precision can be downloaded here: https://zenodo.org/record/6617981/files/resnext50_32x4d_fpn.pth

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