# YOLO v8

This folder contains the script to run YOLO v8 on COCO object detection task.

Variants supplied below for PyTorch and ONNX Runtime in fp32 precision accept input of shape 640x640.

The original documentation of the model is available here: https://docs.ultralytics.com/#ultralytics-yolov8


### Metrics

Based on 1000 images from COCO Dataset for YOLOv8n model in PyTorch framework in fp32 precision

| Metric                  | IoU       | Area   | maxDets |Score  |
|:---:                    |:---:      |:---:   |:---:    |:---:  |
| Average Precision  (AP) |0.50:0.95 |    all | 100 | 0.338 |
| Average Precision  (AP) |0.50      |    all | 100 | 0.452 |
| Average Precision  (AP) |0.75      |    all | 100 | 0.370 |
| Average Precision  (AP) |0.50:0.95 |  small | 100 | 0.122 |
| Average Precision  (AP) |0.50:0.95 | medium | 100 | 0.351 |
| Average Precision  (AP) |0.50:0.95 |  large | 100 | 0.504 |
| Average Recall     (AR) |0.50:0.95 |    all |   1 | 0.265 |
| Average Recall     (AR) |0.50:0.95 |    all |  10 | 0.375 |
| Average Recall     (AR) |0.50:0.95 |    all | 100 | 0.381 |
| Average Recall     (AR) |0.50:0.95 |  small | 100 | 0.133 |
| Average Recall     (AR) |0.50:0.95 | medium | 100 | 0.385 |
| Average Recall     (AR) |0.50:0.95 |  large | 100 | 0.569 |

Based on 1000 images from COCO Dataset for YOLOv8n model in ONNX Runtime framework in fp32 precision

| Metric                  | IoU       | Area   | maxDets |Score  |
|:---:                    |:---:      |:---:   |:---:    |:---:  |
| Average Precision  (AP) |0.50:0.95 |    all | 100 | 0.338|
| Average Precision  (AP) |0.50      |    all | 100 | 0.452|
| Average Precision  (AP) |0.75      |    all | 100 | 0.370|
| Average Precision  (AP) |0.50:0.95 |  small | 100 | 0.122|
| Average Precision  (AP) |0.50:0.95 | medium | 100 | 0.351|
| Average Precision  (AP) |0.50:0.95 |  large | 100 | 0.504|
| Average Recall     (AR) |0.50:0.95 |    all |   1 | 0.265|
| Average Recall     (AR) |0.50:0.95 |    all |  10 | 0.375|
| Average Recall     (AR) |0.50:0.95 |    all | 100 | 0.381|
| Average Recall     (AR) |0.50:0.95 |  small | 100 | 0.133|
| Average Recall     (AR) |0.50:0.95 | medium | 100 | 0.385|
| Average Recall     (AR) |0.50:0.95 |  large | 100 | 0.569|

Based on 1000 images from COCO Dataset for YOLOv8x model in ONNX Runtime framework in fp32 precision

| Metric                  | IoU       | Area   | maxDets |Score  |
|:---:                    |:---:      |:---:   |:---:    |:---:  |
| Average Precision  (AP) |0.50:0.95 |    all | 100 | 0.575|
| Average Precision  (AP) |0.50      |    all | 100 | 0.714|
| Average Precision  (AP) |0.75      |    all | 100 | 0.639|
| Average Precision  (AP) |0.50:0.95 |  small | 100 | 0.336|
| Average Precision  (AP) |0.50:0.95 | medium | 100 | 0.633|
| Average Precision  (AP) |0.50:0.95 |  large | 100 | 0.812|
| Average Recall     (AR) |0.50:0.95 |    all |   1 | 0.409|
| Average Recall     (AR) |0.50:0.95 |    all |  10 | 0.611|
| Average Recall     (AR) |0.50:0.95 |    all | 100 | 0.620|
| Average Recall     (AR) |0.50:0.95 |  small | 100 | 0.361|
| Average Recall     (AR) |0.50:0.95 | medium | 100 | 0.676|
| Average Recall     (AR) |0.50:0.95 |  large | 100 | 0.849|


### Dataset and model

Dataset can be downloaded from here: https://cocodataset.org/#download

PyTorch models in fp32 precision can be downloaded here:
```
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8m.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8l.pt
wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8x.pt
```

You can export a PyTorch model to ONNX Runtime model using the following Python code:

```python
from ultralytics import YOLO
model = YOLO('/path/to/yolov8n.pt')
model.export(format='onnx')
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
python3 run.py -m /path/to/model.pt -p fp32 --framework pytorch
```