# Movenet

This folder contains the script to run Movenet on COCO keypoint prediction task.

Variant supplied below for TFLite accepts input of shape 192x192

The original architecture is available here: https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html

### Metrics

Based on 815 images (i.e., all single person images) from COCO Dataset for TensorFlow framework:

```
Summary: 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.620
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.897
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.692
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.561
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.639
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.673
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.915
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.731
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.617
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.691

                               mean  /      median  / 90th-percentile
 Latency           [ms]:      10.21  /        9.79  /       12.67
 Throughput [samples/s]:      97.94  /      102.14  /       78.95

```

### Dataset and model

Images and annotations can be downloaded from here: https://cocodataset.org/#download
To download images do 
```
wget http://images.cocodataset.org/zips/val2017.zip
```
To download annotations do
```
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
```
Extract the zip file and use ```person_keypoints_val2017.json```

TFLite model in fp32 precision can be downloaded here: https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3?lite-format=tflite or use the following command:
```
wget -q -O movenet_fp32.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3?lite-format=tflite
```

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the model zoo directory,
as well as AIO_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/ampere_model_library
export AIO_NUM_THREADS=1
```

Now you are able to run the ```run.py``` script. 

To get detailed information on the script's recognized arguments run it with -h flag for help.

Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.

Example command for TensorFlow: 

```
python3 run.py -m /path/to/movenet_fp32.tflite  --images_path /path/to/images_path --anno_path /path/to/anno_path --framework tf
```
