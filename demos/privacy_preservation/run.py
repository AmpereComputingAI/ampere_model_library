import argparse
from queue import Queue
from pathlib import Path
import os

import cv2

from video_getter import VideoGetter
from video_writer import VideoWriter
from detector import Detector
from pose import Pose
from postprocessing import Postprocessor

from pipeline import Pipeline

def parse_args():
    parser = argparse.ArgumentParser(description='tf-pose-estimation run')
    parser.add_argument('-v', '--video-path', required=True)
    parser.add_argument('-m', '--model-path', required=True)
    parser.add_argument('-d', '--detection-model-path', required=True)
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    # Get info about the video
    cap = cv2.VideoCapture(args.video_path)
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

    getter = VideoGetter(args.video_path, getter_det_queue, pose_postprocessor_queue, frames)

    pipeline = Pipeline(getter_det_queue, postprocessor_writter_queue, pose_postprocessor_queue, frames, args.detection_model_path, args.model_path)

    os.makedirs("out", exist_ok=True)
    writter = VideoWriter(f"out/{Path(args.video_path).stem}.avi", fps, width, height, postprocessor_writter_queue, frames)

    getter.start()
    pipeline.start()
    writter.start()

    import time
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
