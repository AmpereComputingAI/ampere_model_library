import argparse
import os
import time
from pathlib import Path
from queue import Queue
import sys

import cv2
from flask import Flask, Response, render_template, redirect, request
from werkzeug.utils import secure_filename

from pipeline import Pipeline
from video_getter import VideoGetter
from video_writer import VideoWriter


def parse_args():
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('-m', '--model-path', required=True)
    parser.add_argument('-d', '--detection-model-path', required=True)
    parser.add_argument('-f', '--faces', action='store_true', help='Only blur faces')
    parser.add_argument('-s', '--save', action='store_true', help='Save to file')
    return parser.parse_args()

def get_video_parameters(source):
    cap = cv2.VideoCapture(source)
    fps = cap.get(cv2.CAP_PROP_FPS)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    num_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return fps, height, width, num_frames

def start_demo(getter, pipeline, writer):
    print("Starting")
    start_time = time.time()
    fps, height, width, num_frames = get_video_parameters(getter.src)
    getter.start()
    pipeline.start(num_frames)
    writer.start(getter.src, fps, height, width, num_frames, start_time)

app = Flask(__name__)

@app.route('/')
def index():
    """Video streaming home page."""
    return render_template('index.html')

if __name__ == "__main__":
    args = parse_args()

    getter_det_queue = Queue()
    pose_postprocessor_queue = []
    postprocessor_writer_queue = []
    display_idx = 0

    frames = dict()

    getter = VideoGetter(getter_det_queue, pose_postprocessor_queue, frames)

    pipeline = Pipeline(getter_det_queue, postprocessor_writer_queue, pose_postprocessor_queue, frames, args.detection_model_path, args.model_path, args.faces)

    os.makedirs("out", exist_ok=True)
    writer = VideoWriter(postprocessor_writer_queue, frames, args.save)

    start = time.time() # TODO: Reset the timer when changing source

    @app.route('/video_feed')
    def video_feed():
        #Video streaming route. Put this in the src attribute of an img tag
        return Response(get_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

    @app.route('/reset', methods=['POST', 'GET'])
    def reset():
        global display_idx
        print("RESET")
        if getter.src is None:
            return ('', 204)
        getter.stop()
        pipeline.stop()
        writer.stop()
        time.sleep(1) # TODO: If we reset the queues immediately after calling stop, the functions may still be in the loop and crash
        getter_det_queue = Queue()
        pose_postprocessor_queue = []
        postprocessor_writer_queue = []
        frames.clear()
        display_idx = 0
        getter.reset(getter_det_queue, pose_postprocessor_queue)
        pipeline.reset(getter_det_queue, postprocessor_writer_queue, pose_postprocessor_queue)
        writer.reset(postprocessor_writer_queue)
        start_demo(getter, pipeline, writer)
        return redirect('/')
    
    @app.route('/getfile', methods=['POST'])
    def getfile():
        file = request.files['videoSrc']
        filename = secure_filename(file.filename)
        try:
            file.save(filename)
            getter.src = filename
        except FileNotFoundError:
            return redirect('/')
  
        return redirect('/reset')
    
    @app.route('/webcam', methods=['POST'])
    def use_webcam():
        cap = cv2.VideoCapture(0)
        camera_exists = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) != 0.0
        cap.release()
        if camera_exists:
            getter.src = 0
            return redirect('/reset')
        else:
            return ('', 204)

    @app.route('/faces', methods=['POST'])
    def toggle_faces():
        pipeline.postprocessor.faces = not pipeline.postprocessor.faces
  
        return redirect('/')

    def get_frames():
        global display_idx

        if writer.stopped:
            banner = cv2.imread("static/demo_banner.png")
            _, banner_buffer = cv2.imencode(".png", banner)
            banner_frame = banner_buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + banner_frame + b'\r\n')  # concat frame one by one and show result
        if getter.src == 0:
            display_idx = writer.frame_number
        while not display_idx > writer.last_frame:
            while display_idx not in writer.queue:
                time.sleep(0.01)
            try:
                input_image = cv2.resize(writer.frames[display_idx].blurred, (640, 360)) # Helps with slow network connection
                _, buffer = cv2.imencode('.jpg', input_image, (cv2.IMWRITE_JPEG_QUALITY, 70))
            except AttributeError:
                print("Skipping a frame")
            frame = buffer.tobytes()
            if writer.frames[display_idx].detection_idx == display_idx:
                for i in range(writer.last_deleted_idx, display_idx):
                    writer.frames[i] = None
                    writer.last_deleted_idx = display_idx
            display_idx += 1
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result
        end = time.time()
        print("Press CTRL+C to quit")
        banner = cv2.imread("static/demo_banner.png")
        _, banner_buffer = cv2.imencode(".png", banner)
        banner_frame = banner_buffer.tobytes()
        yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + banner_frame + b'\r\n')  # concat frame one by one and show result

    app.run(host="0.0.0.0", debug= False)
    