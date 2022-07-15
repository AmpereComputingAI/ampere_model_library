# Privacy Preservation Demo

## Models

The demo uses two models, one for object detection and one for pose estimation. You can download them here:
- https://tfhub.dev/tensorflow/lite-model/efficientdet/lite2/detection/default/1
- https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/3

## How to run
`AIO_PROCESS_MODE=0 python3 run.py --model-path lite-model_movenet_singlepose_lightning_3.tflite --detection-model-path lite-model_efficientdet_lite2_detection_default_1.tflite --video-path your_video.mp4 `

The script will use the `your_video.mp4` file and output the results in `out/your_video.avi`.

You can use short versions of the flags, too (`-m`, `-d`, `-v`).

Use `--faces` flag (`-f`) to only pixelate faces.

Use `--show` flag (`-s`) in order to show a window displaying the output of the demo while the program is running. If you don't use this flag, the program will only save the output to a file. 

To make the demo run faster, consider lowering the video resolution:
`ffmpeg -i your_video.mp4 -vf scale=1280:-1 smaller.mp4`