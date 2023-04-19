# YOLO v5

This folder contains the script to run YOLO v5 on COCO object detection task.

Variants supplied below for PyTorch and ONNX Runtime in fp32 precision accept input of shape 640x640.

The original documentation of the model is available here: https://docs.ultralytics.com/yolov5/


### Metrics

Based on 1000 images from COCO Dataset for YOLOv5n model in ONNX Runtime framework in fp32 precision

| Metric                  | IoU       | Area   | maxDets |Score  |
|:---:                    |:---:      |:---:   |:---:    |:---:  |
| Average Precision  (AP) |0.50:0.95 |    all | 100 | |
| Average Precision  (AP) |0.50      |    all | 100 | |
| Average Precision  (AP) |0.75      |    all | 100 | |
| Average Precision  (AP) |0.50:0.95 |  small | 100 | |
| Average Precision  (AP) |0.50:0.95 | medium | 100 | |
| Average Precision  (AP) |0.50:0.95 |  large | 100 | |
| Average Recall     (AR) |0.50:0.95 |    all |   1 | |
| Average Recall     (AR) |0.50:0.95 |    all |  10 | |
| Average Recall     (AR) |0.50:0.95 |    all | 100 | |
| Average Recall     (AR) |0.50:0.95 |  small | 100 | |
| Average Recall     (AR) |0.50:0.95 | medium | 100 | |
| Average Recall     (AR) |0.50:0.95 |  large | 100 | |

Based on 1000 images from COCO Dataset for YOLOv5x model in ONNX Runtime framework in fp32 precision

| Metric                  | IoU       | Area   | maxDets |Score  |
|:---:                    |:---:      |:---:   |:---:    |:---:  |
| Average Precision  (AP) |0.50:0.95 |    all | 100 | |
| Average Precision  (AP) |0.50      |    all | 100 | |
| Average Precision  (AP) |0.75      |    all | 100 | |
| Average Precision  (AP) |0.50:0.95 |  small | 100 | |
| Average Precision  (AP) |0.50:0.95 | medium | 100 | |
| Average Precision  (AP) |0.50:0.95 |  large | 100 | |
| Average Recall     (AR) |0.50:0.95 |    all |   1 | |
| Average Recall     (AR) |0.50:0.95 |    all |  10 | |
| Average Recall     (AR) |0.50:0.95 |    all | 100 | |
| Average Recall     (AR) |0.50:0.95 |  small | 100 | |
| Average Recall     (AR) |0.50:0.95 | medium | 100 | |
| Average Recall     (AR) |0.50:0.95 |  large | 100 | |


### Dataset and model

Dataset can be downloaded from here: https://cocodataset.org/#download

PyTorch models in fp32 precision can be downloaded here:
```
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5n.pt
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5s.pt
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5m.pt
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5l.pt
wget https://github.com/ultralytics/yolov5/releases/download/v7.0/yolov5x.pt
```

You can export a PyTorch model to Torchscript and ONNX models using [this script](https://github.com/ultralytics/yolov5/blob/master/export.py):

```
python export.py --weights yolov5s.pt --include torchscript onnx
```

### Running instructions

Before running any code you should first export the PYTHONPATH variable with path pointing to the model zoo directory,
as well as AIO_NUM_THREADS specifying the number of threads to be used.

```
export PYTHONPATH=/path/to/model_zoo
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

Example command: 

```
python3 run.py -m /path/to/model.onnx -p fp32 --framework ort
```

```
python3 run.py -m /path/to/model.torchscript -p fp32 --framework pytorch
```