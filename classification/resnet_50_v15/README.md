# ResNet-50 v1.5


This folder contains the script to run ResNet-50 v1.5 on ImageNet classification task.\
Variant supplied below in three different precisions accepts input of shape 224x224.

The original paper on the architecture is available here: https://arxiv.org/pdf/1512.03385.pdf

### Accuracy:

Based on 1000 images from ImageNet Validation Dataset

|   | &nbsp;&nbsp;&nbsp;&nbsp; Top-1 Accuracy&nbsp;&nbsp;&nbsp;&nbsp;  |&nbsp;&nbsp;&nbsp;&nbsp; Top-5 Accuracy &nbsp;&nbsp;&nbsp;&nbsp; |
|:---:|:---:|:---:|
| FP32  | 75.0%  | 92.4%  |
| FP16  | 74.9%  | 92.3%  |
| INT8  | 74.8%  | 91.0%  |


### Dataset

Dataset can be acquired here https://www.image-net.org/

model can be downloaded from here:
https://www.tensorflow.org/lite/guide/hosted_models

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


Example command: 

```
python3 run.py -m /path/to/model.pb -p fp32
```
