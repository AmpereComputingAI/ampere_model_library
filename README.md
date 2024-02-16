# Ampere Model Library
AML's goal is to make benchmarking of various AI architectures on Ampere CPUs a pleasurable experience :)

This means we want the library to be quick to set up and to get you numbers you are interested in. On top of that we want the code to be readible and well structured so it's easy to inspect what exactly is being measured. If you feel like we are not exactly there, please let us know right away by raising an [issue](https://github.com/AmpereComputingAI/ampere_model_library/issues/new/choose)! Thank you :)
## AML setup

Visit [our dockerhub](https://hub.docker.com/u/amperecomputingai) for our frameworks selection.


```bash
apt update && apt install -y docker.io git
git clone --recursive https://github.com/AmpereComputingAI/ampere_model_library.git
cd ampere_model_library
docker run --privileged=true -v $PWD/:/aml -it amperecomputingai/pytorch:latest  # we also offer onnxruntime and tensorflow
```

Now you should be inside the docker, to setup AML please run:

```bash
cd /aml
bash setup_deb.sh
source set_env_variables.sh
```


## Examples

Architectures are categorized based on the task they were originally envisioned for. Therefore, you will find ResNet and VGG under computer_vision and BERT under natural_language_processing.
Usual workflow is to first setup AML (see [AML setup](#aml-setup)), source environment variables by running ```source set_env_variables.sh``` and run run.py or similarly named python file in the directory of the achitecture you want to benchmark. Some models require additional setup steps to be completed first, which should be described in their respective directories under README.md files.

### ResNet-50 v1.5
note that the example uses PyTorch - we recommend using Ampere Optimized PyTorch for best results (see [AML setup](#aml-setup))
```bash
source set_env_variables.sh
cd computer_vision/classification/resnet_50_v15
IGNORE_DATASET_LIMITS=1 AIO_IMPLICIT_FP16_TRANSFORM_FILTER=".*" AIO_NUM_THREADS=32 python3 run.py -m resnet50 -p fp32 -b 16 -f pytorch
### the command above will run the model utilizing 32 threads, with batch size of 16
### implicit conversion to FP16 datatype will be applied - you can default to fp32 precision by not setting the AIO_IMPLICIT_FP16_ variable
```

**PSA: you can adjust the level of AIO debug messages by setting AIO_DEBUG_MODE to values in range from 0 to 4 (where 0 is the most peaceful)**

### Whisper tiny EN
note that the example uses PyTorch - we recommend using Ampere Optimized PyTorch for best results (see [AML setup](#aml-setup))
```bash
source set_env_variables.sh
cd speech_recognition/whisper/
AIO_IMPLICIT_FP16_TRANSFORM_FILTER=".*" AIO_NUM_THREADS=32 python3 run.py -m tiny.en
### the command above will run the model utilizing 32 threads
### implicit conversion to FP16 datatype will be applied - you can default to fp32 precision by not setting the AIO_IMPLICIT_FP16_ variable
```

### YOLO v8 large
note that the example uses PyTorch - we recommend using Ampere Optimized PyTorch for best results (see [AML setup](#aml-setup))
```bash
source set_env_variables.sh
cd computer_vision/object_detection/yolo_v8
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
AIO_IMPLICIT_FP16_TRANSFORM_FILTER=".*" AIO_NUM_THREADS=32 python3 run.py -m yolov8l.pt -p fp32 -f pytorch
### the command above will run the model utilizing 32 threads
### implicit conversion to FP16 datatype will be applied - you can default to fp32 precision by not setting the AIO_IMPLICIT_FP16_ variable
```

### BERT large
note that the example uses PyTorch - we recommend using Ampere Optimized PyTorch for best results (see [AML setup](#aml-setup))
```bash
source set_env_variables.sh
cd natural_language_processing/extractive_question_answering/bert_large
wget -O bert_large_mlperf.pt https://zenodo.org/records/3733896/files/model.pytorch?download=1
AIO_IMPLICIT_FP16_TRANSFORM_FILTER=".*" AIO_NUM_THREADS=32 python3 run_mlperf.py -m bert_large_mlperf.pt -p fp32 -f pytorch
### the command above will run the model utilizing 32 threads
### implicit conversion to FP16 datatype will be applied - you can default to fp32 precision by not setting the AIO_IMPLICIT_FP16_ variable
```
