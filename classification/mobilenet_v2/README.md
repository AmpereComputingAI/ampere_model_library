# Mobilenet v2


This folder contains the script for Mobilenet V2. The paper is available to be downloaded from here: \
https://arxiv.org/abs/1704.04861

Model name: Intel(R) Xeon(R) Platinum 8124M CPU @ 3.00GHz

Accuracy

|   | &nbsp;&nbsp;&nbsp;&nbsp; Top-1 Accuracy&nbsp;&nbsp;&nbsp;&nbsp;  |&nbsp;&nbsp;&nbsp;&nbsp; Top-5 Accuracy &nbsp;&nbsp;&nbsp;&nbsp; |
|:---:|:---:|:---:|
| FP32  | 70%  | 91 %  |


## Intel

Latency:

|   | &nbsp;&nbsp;&nbsp;&nbsp; 1 thread&nbsp;&nbsp;&nbsp;&nbsp;  |&nbsp;&nbsp;&nbsp;&nbsp; 18 threads &nbsp;&nbsp;&nbsp;&nbsp; |
|:---:|:---:|:---:|
| FP32  | 11 ms | 5 ms  |

Throughput ( batch size 16 )

|   | &nbsp;&nbsp;&nbsp;&nbsp; 1 thread&nbsp;&nbsp;&nbsp;&nbsp;  |&nbsp;&nbsp;&nbsp;&nbsp; 18 threads &nbsp;&nbsp;&nbsp;&nbsp; |
|:---:|:---:|:---:|
| FP32  | 191.17 ips | 467.31 ips  |




### Datasets


* validation dataset can be downloaded from here: \
  https://www.dropbox.com/s/nxgzz67tpux8wud/ILSVRC2012_onspecta.tar.gz </br> 
  (note: you can unpack them using command in terminal "tar -xvf ./file")
 
   
* validation dataset labels can de downloaded from here:\
  https://www.dropbox.com/s/7ej242ym43v635i/imagenet_labels_onspecta.txt

* model is available to download from here:\
  https://www.dropbox.com/s/jnop89eowak1w6n/mobilenet_v2_tf_fp32.pb

### Run instructions

To run the model you have to first specify a few things. For best experience you can export environment variables 

For linux & Mac

```
export OMP_NUM_THREADS=1
export PYTHONPATH=/path/to/model_zoo
export IMAGENET_IMG_PATH=/path/to/images
export IMAGENET_LABELS_PATH=/path/to/labels
```

For Windows:

```
SET OMP_NUM_THREADS=1
SET PYTHONPATH=\path\to\model_zoo
SET IMAGENET_IMG_PATH=\path\to\images
SET IMAGENET_LABELS_PATH=\path\to\labels
```

next you will be able to run the run.py script. 

to get detailed information about the script run it with -h flag for help.

the path to model (with a flag "-m") as well as it's precision (with a flag "-p") have to be specified\
Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.


Example command: 

```
python3 run.py -m /path/to/model.pb -p fp32
```



  