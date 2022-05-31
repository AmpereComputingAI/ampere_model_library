# Privacy Preservation Demo

## Models

The demo uses two models, one for object detection and one for pose estimation. You can download them here:
- https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/default/1
- https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3

## How to run
`AIO_PROCESS_MODE=0 python3 run.py --model-path lite-model_movenet_singlepose_lightning_3.tflite --detection-model-path lite-model_efficientdet_lite2_detection_default_1.tflite --video-path your_video.mp4 `

The script will use the `your_video.mp4` file and output the results in `out/your_video.avi`.

To make the demo run faster, consider lowering the video resolution:
`ffmpeg -i your_video.mp4 -vf scale=1280:-1 smaller.mp4`