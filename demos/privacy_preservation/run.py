import argparse
import os
import time
from pathlib import Path
from queue import Queue

import cv2
from flask import Flask, Response, render_template

from pipeline import Pipeline
from video_getter import VideoGetter
from video_writer import VideoWriter


def parse_args():
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('-v', '--video-path')
    parser.add_argument('-m', '--model-path', required=True)
    parser.add_argument('-d', '--detection-model-path', required=True)
    parser.add_argument('-f', '--faces', action='store_true', help='Only blur faces')
    parser.add_argument('-s', '--save', action='store_true', help='Save to file')
    parser.add_argument('-c', '--camera', action='store_true', help='Use webcam')
    return parser.parse_args()

app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == "__main__":
    args = parse_args()

    if args.camera:
        source = 0
        out_path = "out/camera.avi"
    else:
        source = args.video_path
        out_path = f"out/{Path(args.video_path).stem}.avi"

    # Get info about the video
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    getter_det_queue = Queue()
    det_pose_queue = Queue()
    # pose_postprocessor_queue = Queue()
    pose_postprocessor_queue = []
    # postprocessor_writer_queue = Queue()
    postprocessor_writer_queue = []

    frames = dict()

    getter = VideoGetter(source, getter_det_queue, pose_postprocessor_queue, frames)

    pipeline = Pipeline(getter_det_queue, postprocessor_writer_queue, pose_postprocessor_queue, frames, args.detection_model_path, args.model_path, args.faces)

    os.makedirs("out", exist_ok=True)
    writer = VideoWriter(out_path, fps, width, height, postprocessor_writer_queue, frames, args.save, num_frames)

    start = time.time()
    getter.start()
    pipeline.start()
    writer.start()

    @app.route('/video_feed')
    def video_feed():
        #Video streaming route. Put this in the src attribute of an img tag
        return Response(get_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def get_frames():
        if args.camera:
            idx = writer.frame_number
        else:
            idx = 0
        while not idx > writer.last_frame:
            while idx not in writer.queue:
                time.sleep(0.01)
            try:
                ret, buffer = cv2.imencode('.jpg', writer.frames[idx].blurred)
            except AttributeError:
                print("Skipping a frame") # TODO: This happens because of removing the frames in video_writer.py
            frame = buffer.tobytes()
            idx += 1
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        end = time.time()
        print("Total time: ", end - start)
        print("FPS: ", (writer.last_frame + 1) / (end - start))
        print("Press CTRL+C to quit")

    app.run(host="0.0.0.0", debug= False)
    