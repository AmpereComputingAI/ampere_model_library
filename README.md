# Ampere Model Library
Model Library goal is to provide facilities necessary for:
- benchmarking various AI architectures easily with different frameworks 
- testing accuracy of AI models on data representative of their envisioned task
- aiding comparison of and experiments on various available AI architectures

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

To run model of choice go to its directory, eg. computer_vision/classification/resnet_50_v15, and follow the instructions supplied there.
