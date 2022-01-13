# Resnet 50 v1


This folder contains the script to run Resnet 50 on ImageNet classification task in pytorch and onnx runtime framework.\
Variant supplied below in fp32 precision accepts input of shape 224x224.

The original paper on the architecture is available here: https://arxiv.org/abs/1512.03385


### Accuracy:

based on 1000 images from ImageNet Validation Dataset

|   | &nbsp;&nbsp;&nbsp;&nbsp; Top-1 Accuracy&nbsp;&nbsp;&nbsp;&nbsp;  |&nbsp;&nbsp;&nbsp;&nbsp; Top-5 Accuracy &nbsp;&nbsp;&nbsp;&nbsp; |
|:---:|:---:|:---:|
| PyTorch FP32  | 72.2%  | 91.0 %  |
| ONNX RT FP32  | 75.0%  | 92.4 %  |



### Dataset

Dataset can be downloaded here: https://www.image-net.org/

ONNX model can be downloaded here: https://zenodo.org/record/2592612


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

The precision (with a flag "-p") as well as framework (with a flag "--framework") have to be specified.\
Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.


Example command for PyTorch: 

```
python3 run.py -p fp32 --framework pytorch
```

Example command for ONNX RT: 

```
python3 run.py -m /path/to/model.onnx -p fp32 --framework ort
```