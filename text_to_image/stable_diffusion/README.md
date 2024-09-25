# Stable Diffusion and Stable Diffusion XL


This folder contains the script to run Stable Diffusion in PyTorch framework using randomized prompts. 


### Model:

SD Model details are available here: https://huggingface.co/stabilityai/stable-diffusion-2-1-base \
SD Model can be downloaded here: [https://huggingface.co/stabilityai/stable-diffusion-2-1-base/blob/main/v2-1_512-ema-pruned.ckpt](https://huggingface.co/stabilityai/stable-diffusion-2-1-base/resolve/main/v2-1_512-ema-pruned.ckpt)

SD-XL Model details are available here: https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0 

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the Ampere Model Library directory,
as well as AIO_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/ampere_model_library
export AIO_NUM_THREADS=1
```

Now you are able to run the run.py script. 

To get detailed information on the script's recognized arguments run it with -h flag for help. Please remember to specify 
config file - you can use the file in stablediffusion folder in this directory under stablediffusion/configs/stable-diffusion/intel/v2-inference-fp32.yaml 

Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.

Example command for Stable Diffusion: 

```
AIO_NUM_THREADS=128 python3 run.py -m /path/to/model/v2-1_512-ema-pruned.ckpt --config stablediffusion/configs/stable-diffusion/intel/v2-inference-fp32.yaml
```

Example command for Stable Diffusion XL:

```
AIO_NUM_THREADS=16 python3 text_to_image/stable_diffusion/run_hf.py -m stabilityai/stable-diffusion-xl-base-1.0 -p fp32
```

if you want to run the model on Altra with AIO software in order to achieve the best performance use this command:
```
AIO_NUM_THREADS=128 AIO_MERGE_LINEAR=0 OPENBLAS_NUM_THREADS=10 AIO_IMPLICIT_FP16_TRANSFORM_FILTER=".*" python text_to_image/stable_diffusion/run_hf.py -m stabilityai/stable-diffusion-xl-base-1.0 -p fp32
```
AIO_MERGE_LINEAR=0 allows to compile decoder of the base model with aio \
OPENBLAS_NUM_THREADS=10 sets the number of threads OpenBLAS will use for efficient linear algebra computations \
AIO_IMPLICIT_FP16_TRANSFORM_FILTER=".*" will transform fp32 operations to fp16 operations on the fly.
