# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import os
import time
from queue import Queue
from threading import Thread

from detector import Detector
from pose import Pose
from postprocessing import Postprocessor


class Pipeline:
    def __init__(self, getter_queue, writer_queue, pose_postprocessor_queue, frames, detection_model_path, model_path, faces):
        self.getter_queue = getter_queue
        self.writer_queue = writer_queue
        self.stopped = True
        self.frames = frames
        self.det_pose_queue = Queue()
        self.pose_postprocessor_queue = pose_postprocessor_queue

        try:
            omp_num_threads = int(os.environ["OMP_NUM_THREADS"])
        except KeyError:
            omp_num_threads = None

        self.det = Detector(detection_model_path, omp_num_threads, self.getter_queue, self.det_pose_queue, self.frames)
        self.pose_estimator = Pose(model_path, omp_num_threads, self.det_pose_queue, self.pose_postprocessor_queue, frames)
        self.postprocessor = Postprocessor(self.pose_postprocessor_queue, self.writer_queue, frames, faces)

    
    def start(self, last_frame):
        self.stopped = False
        Thread(target=self.process, args=(last_frame,)).start()
        return self

    def process(self, last_frame):
        self.det.start()
        self.pose_estimator.start()
        self.postprocessor.start(last_frame)

        self.stopped = False
        while not self.postprocessor.stopped:
            time.sleep(0.1) # Without it, this thread blocks the other threads
        self.stop()
    
    def stop(self):
        self.stopped = True
        self.det.stop()
        self.pose_estimator.stop()
        self.postprocessor.stop()
    
    def reset(self, getter_queue, writer_queue, pose_postprocessor_queue):
        self.getter_queue = getter_queue
        self.writer_queue = writer_queue
        self.det_pose_queue = Queue()
        self.pose_postprocessor_queue = pose_postprocessor_queue
        self.det.reset(self.getter_queue, self.det_pose_queue)
        self.pose_estimator.reset(self.det_pose_queue, pose_postprocessor_queue)
        self.postprocessor.reset(pose_postprocessor_queue, self.writer_queue)