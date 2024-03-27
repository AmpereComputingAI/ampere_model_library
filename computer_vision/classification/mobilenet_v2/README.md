# MobileNet v2


This folder contains the script to run Inception v3 on ImageNet classification task 
in TensorFlow, PyTorch and ONNX Runtime framework.

Variant supplied below in three different precisions accepts input of shape 224x224 and has 1.0x multiplier.

Variant supplied below for PyTorch framework in fp32 precision accepts input of shape 224x224

Variant supplied below for ONNX Runtime framework in two different precisions accepts input of shape 224x224.

The original paper on the architecture is available here: https://arxiv.org/pdf/1801.04381


### Accuracy:

based on 1000 images from ImageNet Validation Dataset for TensorFlow framework

|   | &nbsp;&nbsp;&nbsp;&nbsp; Top-1 Accuracy&nbsp;&nbsp;&nbsp;&nbsp;  |&nbsp;&nbsp;&nbsp;&nbsp; Top-5 Accuracy &nbsp;&nbsp;&nbsp;&nbsp; |
|:---:|:---:|:---:|
| FP32  | 70%  | 90.5 %  |
| FP16  | 70%  | 90.5 %  |
| INT8  | 69%  | 90.3 %  |

based on 1000 images from ImageNet Validation Dataset for PyTorch framework

|   | &nbsp;&nbsp;&nbsp;&nbsp; Top-1 Accuracy&nbsp;&nbsp;&nbsp;&nbsp;  |&nbsp;&nbsp;&nbsp;&nbsp; Top-5 Accuracy &nbsp;&nbsp;&nbsp;&nbsp; |
|:---:|:---:|:---:|
| FP32  | 67.4 %  | 87.9 %  |

based on 1000 images from ImageNet Validation Dataset for ONNX Runtime framework

|   | &nbsp;&nbsp;&nbsp;&nbsp; Top-1 Accuracy&nbsp;&nbsp;&nbsp;&nbsp;  |&nbsp;&nbsp;&nbsp;&nbsp; Top-5 Accuracy &nbsp;&nbsp;&nbsp;&nbsp; |
|:---:|:---:|:---:|
| FP32  | 69.5%  | 88.6%  |
| FP16  | 69.9%  | 90.4%  |

### Dataset and models

Dataset can be downloaded here: https://www.image-net.org/

TensorFlow models can be downloaded here: https://www.tensorflow.org/lite/guide/hosted_models

ONNX Runtime model in fp32 precision can be downloaded here: https://s3.amazonaws.com/onnx-model-zoo/mobilenet/mobilenetv2-1.0/mobilenetv2-1.0.tar.gz

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

For Tensorflow implementation the path to model (with a flag "-m") as well as its precision (with a flag "-p") \ 
have to be specified. Please note that the default batch size is 1 and if not specified otherwise the script will \ 
run for 1 minute.

For PyTorch implementation the precision (with a flag "-p") as well as framework (with a fla "--framework") have to be specified.\

For ONNX Runtime implementation the path to model (with a flag "-m") as well as its precision (with a flag "-p") have to be specified.

Example command for TensorFlow: 

```
python3 run.py -m /path/to/model.pb -p fp32
```

Example command for PyTorch:

```
python3 run.py -p fp32 --framework pytorch
```

Example command for ONNX Runtime: 

```
python3 run.py -m /path/to/model.onnx -p fp32 --framework ort
```
