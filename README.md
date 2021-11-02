# model_zoo
Model zoo goal is to provide facilities necessary for:
- benchmarking various AI architectures easily with different frameworks 
- testing accuracy of AI models on data representative of their envisioned task
- aiding comparison of and experiments on various available AI architectures

## setup

```bash
git clone git@github.com:OnSpecta/model_zoo.git
cd model_zoo
git submodule update --init --recursive
```

To run model of choice go to its directory, eg. computer_vision/classification/resnet_50_v15, and follow the instructions supplied there.
Alternatively use dev wrapper: https://github.com/OnSpecta/model_zoo_dev
