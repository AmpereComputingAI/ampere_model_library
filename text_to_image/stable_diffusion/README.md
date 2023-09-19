# Stable Diffusion


This folder contains the script to run Stable Diffusion in PyTorch framework using randomized prompts. 


### Model:

Model details are available here: https://huggingface.co/stabilityai/stable-diffusion-2-1-base \
Model can be downloaded here: https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.ckpt


### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the model zoo directory,
as well as DLS_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/model_zoo
export AIO_NUM_THREADS=1
```

Now you are able to run the run.py script. 

To get detailed information on the script's recognized arguments run it with -h flag for help. Please remember to specify 
config file - you can use the file in stablediffusion folder in this directory under stablediffusion/configs/stable-diffusion/intel/v2-inference-fp32.yaml 

Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.

Example command: 

```
AIO_NUM_THREADS=128 python3 run.py -m /path/to/model/v2-1_512-ema-pruned.ckpt --config stablediffusion/configs/stable-diffusion/intel/v2-inference-fp32.yaml
```