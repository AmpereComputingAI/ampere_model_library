import time
from threading import Thread

import cv2

from frame import Frame


class VideoGetter:
    """
    Class that continuously gets frames from a VideoCapture object using a dedicated thread.
    """

    def __init__(self, det_queue, postprocessor_queue, frames):
        self.src = None
        self.stopped = False
        self.det_queue = det_queue
        self.postprocessor_queue = postprocessor_queue
        self.idx = 0
        self.frames = frames
        self.time_since_last_detection = 0.0
        self.last_detection_idx = 0

    def start(self):
        self.stream = cv2.VideoCapture(self.src)
        (self.grabbed, self.frame) = self.stream.read()
        self.stopped = False
        Thread(target=self.get, args=()).start()
        return self

    def get(self):
        while not self.stopped:
            (self.grabbed, self.frame) = self.stream.read()
            if not self.grabbed:
                for i in range(8): # That's not the ideal way of doing it
                    self.det_queue.put(None)
                self.stop()
                break
            self.frames[self.idx] = Frame(self.frame)
            # if self.idx % 5 == 0:
            if time.time() - self.time_since_last_detection >= 0.2 or self.idx - self.last_detection_idx >= 5:
                self.time_since_last_detection = time.time()
                self.frames[self.idx].detection_idx = self.idx
                self.last_detection_idx = self.idx
                self.det_queue.put(self.idx)
            else:
                self.frames[self.idx].detection_idx = self.last_detection_idx
                self.postprocessor_queue.append(self.idx)
            # print("Getter", self.idx)
            self.idx += 1

    def stop(self):
        self.stopped = True
        try:
            self.stream.release()
        except AttributeError:
            pass
    
    def reset(self, det_queue, postprocessor_queue):
        self.det_queue = det_queue
        self.postprocessor_queue = postprocessor_queue
        self.idx = 0
        self.time_since_last_detection = 0.0
        self.last_detection_idx = 0
