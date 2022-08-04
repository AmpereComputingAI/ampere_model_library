import argparse
from queue import Queue
from pathlib import Path
import os
import time

import cv2
from flask import Flask, render_template, Response

from video_getter import VideoGetter
from video_writer import VideoWriter
from detector import Detector
from pose import Pose
from postprocessing import Postprocessor

from pipeline import Pipeline

def parse_args():
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('-v', '--video-path')
    parser.add_argument('-m', '--model-path', required=True)
    parser.add_argument('-d', '--detection-model-path', required=True)
    parser.add_argument('-f', '--faces', action='store_true', help='Only blur faces')
    parser.add_argument('-s', '--show', action='store_true', help='Show window displaying the demo, otherwise only save to file')
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
    cap.release()

    getter_det_queue = Queue()
    det_pose_queue = Queue()
    # pose_postprocessor_queue = Queue()
    pose_postprocessor_queue = []
    # postprocessor_writter_queue = Queue()
    postprocessor_writter_queue = []

    frames = dict()

    getter = VideoGetter(source, getter_det_queue, pose_postprocessor_queue, frames)

    pipeline = Pipeline(getter_det_queue, postprocessor_writter_queue, pose_postprocessor_queue, frames, args.detection_model_path, args.model_path, args.faces)

    os.makedirs("out", exist_ok=True)
    writter = VideoWriter(out_path, fps, width, height, postprocessor_writter_queue, frames, args.show)

    getter.start()
    pipeline.start()
    writter.start()

    @app.route('/video_feed')
    def video_feed():
        #Video streaming route. Put this in the src attribute of an img tag
        return Response(get_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    def get_frames():
        while True:
            idx = writter.frame_number

            while idx not in writter.queue:
                time.sleep(0.02)
            ret, buffer = cv2.imencode('.jpg', writter.frames[idx].blurred)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

    app.run(host="0.0.0.0", debug=False)
    
    start = time.time()

    still_running = True

    # Check if all frames were processed
    while still_running:
        still_running = False
        if not pipeline.stopped:
            still_running = True
        time.sleep(0.1) # Without it, this thread blocks the other threads
    
    # Wait until all frames are written
    writter.last_frame = max(writter.queue)
    while not writter.stopped:
        time.sleep(0.1)

    end = time.time()
    print("Total time: ", end - start)
    print("FPS: ", (writter.last_frame + 1) / (end - start))
