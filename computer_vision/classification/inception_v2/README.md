# Inception v2


This folder contains the script to run Inception v2 on ImageNet classification task.\
Variant supplied below for TensorFlow in three different precisions accepts input of shape 224x224.

Variant supplied below for ONNX Runtime in fp16 precision accepts input of shape 224x224.

The original paper on the architecture is available here: https://arxiv.org/abs/1512.00567


### Accuracy:

Based on 1000 images from ImageNet Validation Dataset for TensorFlow framework

|   | &nbsp;&nbsp;&nbsp;&nbsp; Top-1 Accuracy&nbsp;&nbsp;&nbsp;&nbsp;  |&nbsp;&nbsp;&nbsp;&nbsp; Top-5 Accuracy &nbsp;&nbsp;&nbsp;&nbsp; |
|:---:|:---:|:---:|
| FP32  | 73.0%  | 90.8%  |
| FP16  | 72.8%  | 90.9%  |
| INT8  | 72.3%  | 90.3%  |

Based on 1000 images from ImageNet Validation Dataset for ONNX Runtime framework

|   | &nbsp;&nbsp;&nbsp;&nbsp; Top-1 Accuracy&nbsp;&nbsp;&nbsp;&nbsp;  |&nbsp;&nbsp;&nbsp;&nbsp; Top-5 Accuracy &nbsp;&nbsp;&nbsp;&nbsp; |
|:---:|:---:|:---:|
| FP16  | 73.0% | 90.8%  |

### Dataset and models

Dataset can be downloaded from here: https://www.image-net.org/

ONNX Runtime model in fp16 precision can be downloaded here: censored due to licensing

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the model zoo directory,
as well as DLS_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/model_zoo
export DLS_NUM_THREADS=1
```

For the best experience we also recommend setting environment variables as specified below.

```
export IMAGENET_IMG_PATH=/path/to/images
export IMAGENET_LABELS_PATH=/path/to/labels
```

Now you are able to run the run.py script. 

To get detailed information on the script's recognized arguments run it with -h flag for help.

The path to model (with a flag "-m") as well as its precision (with a flag "-p") have to be specified.\
Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.


Example command for TensorFlow: 

```
python3 run.py -m /path/to/model.pb -p fp32 --framework tf
```

Example command for ONNX Runtime: 

```
python3 run.py -m /path/to/model.onnx -p fp16 --framework ort
```
