# Ampere Model Library
Model Library goal is to provide facilities necessary for:
- benchmarking various AI architectures easily with different frameworks 
- testing accuracy of AI models on data representative of their envisioned task
- aiding comparison of and experiments on various available AI architectures

## Ampere-optimized framework setup

Visit https://solutions.amperecomputing.com/solutions/ampere-ai to obtain the latest available Docker image for your framework of choice.

Go to Downloads section and click on EULA and Docker Image, then after accepting EULA and obtaining your unique download URL:

```bash
apt update && apt install docker.io
apt-get update && apt-get install wget
wget -O ampere_framework.tar.gz “<your_unique_url>”
docker load < ampere_framework.tar.gz
```

When the docker load completes you will see the name of the image. Please supply it under AML setup as ampere_framework_image. 

You can also check available images on the system by running:

```bash
docker image ls -a
```

## AML setup

```bash
git clone git@github.com:AmpereComputingAI/ampere_model_library.git
cd ampere_model_library
git submodule update --init --recursive
docker run --privileged=true --name ampere_framework -v ./:/aml -it ampere_framework_image
```

Now you should be inside the docker, to setup AML please run:

```bash
cd /aml
bash setup_deb.sh
source set_env_variables.sh
```

To run model of choice go to its directory, eg. computer_vision/classification/resnet_50_v15, and follow the instructions supplied there.
