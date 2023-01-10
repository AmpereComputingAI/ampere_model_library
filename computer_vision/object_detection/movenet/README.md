# Movenet

This folder contains the script to run Movenet on COCO keypoint prediction task.

Variant supplied below for TFLite accepts input of shape 192x192

The original architecture is available here: https://blog.tensorflow.org/2021/05/next-generation-pose-detection-with-movenet-and-tensorflowjs.html

### Metrics

Based on 50 images from COCO Dataset for TensorFlow framework in int8 precision

Summary: 
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.107
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.175
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.118
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.082
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.143
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] = 0.109
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] = 0.175
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] = 0.121
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = 0.082
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] = 0.143

                               mean  /      median  / 90th-percentile
 Latency           [ms]:      33.24  /       32.44  /       37.71
 Throughput [samples/s]:      30.08  /       30.83  /       26.52

### Dataset and model

Dataset can be downloaded from here: https://cocodataset.org/#download

TFLite model in fp32 precision can be downloaded here: https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3

```
wget -O lite-model_efficientdet_lite2_detection_default_1.tflite https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/default/1?lite-format=tflite
```

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the model zoo directory,
as well as AIO_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/ampere_model_library
export AIO_NUM_THREADS=1
```

For the best experience we also recommend setting environment variables as specified below.

```
export COCO_IMG_PATH=/path/to/images
export COCO_ANNO_PATH=/path/to/annotations
```

Now you are able to run the run.py script. 

To get detailed information on the script's recognized arguments run it with -h flag for help.

The path to model (with a flag "-m") as well as its precision (with a flag "-p") have to be specified.

Please note that the default batch size is 1 and if not specified otherwise the script will run for 1 minute.

Example command for TensorFlow: 

```
python3 run.py -m /path/to/model.pb -p int8 --framework tf
```
