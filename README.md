![Ampere AI](https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/ampere_logo_®_primary_stacked_rgb.png "Ampere AI")
# Ampere Model Library
![CI tests](https://github.com/github/docs/actions/workflows/test.yml/badge.svg)
![PyTorch pull count](https://img.shields.io/docker/pulls/amperecomputingai/pytorch?logo=pytorch&label=PyTorch&labelColor=%23ffc9bb&color=%23ffa590&link=https%3A%2F%2Fhub.docker.com%2Fr%2Famperecomputingai%2Fpytorch)
![TF pull count](https://img.shields.io/docker/pulls/amperecomputingai/tensorflow?logo=tensorflow&label=TensorFlow&labelColor=%23e6cc00&color=%23e69b00&link=https%3A%2F%2Fhub.docker.com%2Fr%2Famperecomputingai%2Ftensorflow)
![ORT pull count](https://img.shields.io/docker/pulls/amperecomputingai/onnxruntime?logo=onnx&logoColor=black&label=ONNXRT&labelColor=%23e5e5e5&color=%23cccccc&link=https%3A%2F%2Fhub.docker.com%2Fr%2Famperecomputingai%2Fonnxruntime)
![llama.cpp pull count](https://img.shields.io/docker/pulls/amperecomputingai/llama.cpp?logo=meta&logoColor=black&label=llama.cpp&labelColor=violet&color=purple)

AML's goal is to make benchmarking of various AI architectures on Ampere CPUs a pleasurable experience :)

This means we want the library to be quick to set up and to get you numbers you are interested in. On top of that we want the code to be readible and well structured so it's easy to inspect what exactly is being measured. If you feel like we are not exactly there, please let us know right away by raising an [issue](https://github.com/AmpereComputingAI/ampere_model_library/issues/new/choose)! Thank you :)
## AML setup
![Ampere AI solutions](https://uawartifacts.blob.core.windows.net/upload-files/ai_infographic_cloud_47da3198d8.jpg "Ampere AI solutions")
Visit [our dockerhub](https://hub.docker.com/u/amperecomputingai) for our frameworks selection.


```bash
sudo apt update && sudo apt install -y docker.io
sudo docker run --privileged=true -it amperecomputingai/pytorch:latest
# we also offer onnxruntime and tensorflow
```
You should see terminal output similar to that one:

![Ampere docker welcome prompt](https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/Screenshot+2024-02-16+at+20.16.37.png "Ampere docker welcome prompt")

Now, inside the Docker container, run:

```bash
git clone --recursive https://github.com/AmpereComputingAI/ampere_model_library.git
cd ampere_model_library
bash setup_deb.sh
source set_env_variables.sh
```

You are good to go! 👌


## Examples

### The go-to solution is benchmark.py script
Benchmark script allows you to quickly evaluate performance of your Ampere system on the example of:
- ResNet-50 v1.5
- Whisper medium EN
- DLRM
- BERT large
- YOLO v8s

It's incredibly user-friendly and designed to assist you in getting the best out of your system.

**After completing setup with Ampere Optimized PyTorch (see [AML setup](#aml-setup)), it's as easy as:**
```bash
python3 benchmark.py --no-interactive  # remove --no-interactive if you want a quick estimation of performance
```

![Evaluation results](https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/Screenshot+2024-03-01+at+19.53.08.png "Evaluation results")

### Running particular AI architectures

Architectures are categorized based on the task they were originally envisioned for. Therefore, you will find ResNet and VGG under computer_vision and BERT under natural_language_processing.
Usual workflow is to first setup AML (see [AML setup](#aml-setup)), source environment variables by running ```source set_env_variables.sh``` and run run.py or similarly named python file in the directory of the achitecture you want to benchmark. Some models require additional setup steps to be completed first, which should be described in their respective directories under README.md files.

### ResNet-50 v1.5
![ResNet-50 architecture](https://miro.medium.com/v2/resize:fit:720/format:webp/0*tH9evuOFqk8F41FG.png "ResNet-50 architecture")

Note that the example uses PyTorch - we recommend using Ampere Optimized PyTorch for best results (see [AML setup](#aml-setup)).
```bash
source set_env_variables.sh
IGNORE_DATASET_LIMITS=1 AIO_IMPLICIT_FP16_TRANSFORM_FILTER=".*" AIO_NUM_THREADS=32 python3 computer_vision/classification/resnet_50_v15/run.py -m resnet50 -p fp32 -b 16 -f pytorch
```
The command above will run the model utilizing 32 threads, with batch size of 16. Implicit conversion to FP16 datatype will be applied - you can default to fp32 precision by not setting the **AIO_IMPLICIT_FP16_TRANSFORM_FILTER** variable.

**PSA: you can adjust the level of AIO debug messages by setting AIO_DEBUG_MODE to values in range from 0 to 4 (where 0 is the most peaceful)**

### Whisper tiny EN
![Whisper architecture](https://raw.githubusercontent.com/openai/whisper/main/approach.png "Whisper architecture")

Note that the example uses PyTorch - we recommend using Ampere Optimized PyTorch for best results (see [AML setup](#aml-setup)).
```bash
source set_env_variables.sh
AIO_IMPLICIT_FP16_TRANSFORM_FILTER=".*" AIO_NUM_THREADS=32 python3 speech_recognition/whisper/run.py -m tiny.en
```
The command above will run the model utilizing 32 threads, implicit conversion to FP16 datatype will be applied - you can default to fp32 precision by not setting the **AIO_IMPLICIT_FP16_TRANSFORM_FILTER** variable.

### LLaMA2 7B
![Transformer vs LLaMA](https://miro.medium.com/v2/resize:fit:1400/1*g9cykAlrYrNkG-rVTIKQ2Q.png "https://www.youtube.com/shorts/A6LOVMymJhs")

Note that the example uses PyTorch - we recommend using Ampere Optimized PyTorch for best results (see [AML setup](#aml-setup)).

**Before running this example you need to be granted access by Meta to LLaMA2 model. Go here: [Meta](https://ai.meta.com/resources/models-and-libraries/llama-downloads) and here: [HF](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) to learn more.**
```bash
source set_env_variables.sh
wget https://github.com/tloen/alpaca-lora/raw/main/alpaca_data.json
AIO_IMPLICIT_FP16_TRANSFORM_FILTER=".*" AIO_NUM_THREADS=32 python3 natural_language_processing/text_generation/llama2/run.py -m meta-llama/Llama-2-7b-chat-hf --dataset_path=alpaca_data.json
```
The command above will run the model utilizing 32 threads, implicit conversion to FP16 datatype will be applied - you can default to fp32 precision by not setting the **AIO_IMPLICIT_FP16_TRANSFORM_FILTER** variable.

### YOLO v8 large
![YOLO object detection](https://miro.medium.com/v2/resize:fit:1358/1*r_3a2KsqTznF4Pt-MnF00Q.jpeg "YOLO object detection")

Note that the example uses PyTorch - we recommend using Ampere Optimized PyTorch for best results (see [AML setup](#aml-setup)).
```bash
source set_env_variables.sh
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
AIO_IMPLICIT_FP16_TRANSFORM_FILTER=".*" AIO_NUM_THREADS=32 python3 computer_vision/object_detection/yolo_v8/run.py -m yolov8l.pt -p fp32 -f pytorch
```
The command above will run the model utilizing 32 threads, implicit conversion to FP16 datatype will be applied - you can default to fp32 precision by not setting the **AIO_IMPLICIT_FP16_TRANSFORM_FILTER** variable.

### BERT large
![BERT embeddings](https://miro.medium.com/v2/resize:fit:1400/0*m_kXt3uqZH9e7H4w.png "BERT embeddings")

Note that the example uses PyTorch - we recommend using Ampere Optimized PyTorch for best results (see [AML setup](#aml-setup)).
```bash
source set_env_variables.sh
wget -O bert_large_mlperf.pt https://zenodo.org/records/3733896/files/model.pytorch?download=1
AIO_IMPLICIT_FP16_TRANSFORM_FILTER=".*" AIO_NUM_THREADS=32 python3 natural_language_processing/extractive_question_answering/bert_large/run_mlperf.py -m bert_large_mlperf.pt -p fp32 -f pytorch
```
The command above will run the model utilizing 32 threads, implicit conversion to FP16 datatype will be applied - you can default to fp32 precision by not setting the **AIO_IMPLICIT_FP16_TRANSFORM_FILTER** variable.
