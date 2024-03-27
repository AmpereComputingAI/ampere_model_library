# 3D-Unet BraTS 19


This folder contains the script to run 3D-Unet on MLPerf BraTS 2019 brain tumor segmentation task in Tensorflow and PyTorch framework.

The original paper on the architecture is available here: https://arxiv.org/pdf/1606.06650.pdf

### Accuracy:

Based on 10 images for PyTorch framework

|   | &nbsp;&nbsp;&nbsp;&nbsp; Mean whole tumor&nbsp;&nbsp;&nbsp;&nbsp;  |&nbsp;&nbsp;&nbsp;&nbsp; Mean tumor core &nbsp;&nbsp;&nbsp;&nbsp; |&nbsp;&nbsp;&nbsp;&nbsp; Mean enhancing tumor &nbsp;&nbsp;&nbsp;&nbsp; |&nbsp;&nbsp;&nbsp;&nbsp; Mean composite &nbsp;&nbsp;&nbsp;&nbsp; |
|:---:|:---:|:---:|:---:|:---:|
| FP32  | 88.2%  | 93.2%  | 85.2%  | 88.9%  |

Based on 10 images for PyTorch framework

|   | &nbsp;&nbsp;&nbsp;&nbsp; Mean whole tumor&nbsp;&nbsp;&nbsp;&nbsp;  |&nbsp;&nbsp;&nbsp;&nbsp; Mean tumor core &nbsp;&nbsp;&nbsp;&nbsp; |&nbsp;&nbsp;&nbsp;&nbsp; Mean enhancing tumor &nbsp;&nbsp;&nbsp;&nbsp; |&nbsp;&nbsp;&nbsp;&nbsp; Mean composite &nbsp;&nbsp;&nbsp;&nbsp; |
|:---:|:---:|:---:|:---:|:---:|
| FP32  | 87.6%  | 92.0%  | 84.8%  | 88.2%  |

### Dataset and models

Dataset can be downloaded here: censored due to licensing

Extract the dataset:
```
tar -xvf brats_19.tar.gz
```

TensorFlow model can be downloaded here: https://zenodo.org/record/3928991/files/224_224_160.pb

PyTorch model can be downloaded here: https://zenodo.org/record/3904106/files/fold_1.zip

Unzip the model:
```
unzip fold_1.zip
```

The model you need to point to when running the script can be found under fold_1/nnUNet/3d_fullres/Task043_BraTS2019/nnUNetTrainerV2__nnUNetPlansv2.mlperf.1/fold_1/model_best.model

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the model zoo directory,
as well as AIO_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/model_zoo
export AIO_NUM_THREADS=1
```

For the best experience we also recommend setting environment variables as specified below.

```
export BRATS19_DATASET_PATH=/path/to/dataset
```

Now you are able to run the run.py script.

To get detailed information on the script's recognized arguments run it with -h flag for help.

The precision (with a flag "-p") as well as framework (with a flag "--framework") have to be specified.
Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.

Example command for TensorFlow:

```
python3 run.py -m path/to/model.pb -p fp32 --framework tf
```

Example command for PyTorch:

```
python3 run.py -m path/to/model.model -p fp32 --framework pytorch
```
