from threading import Thread
import time
import cv2
import numpy as np
import tensorflow as tf


class Detector:
    """
    Class that continuously detects objects on image using a dedicated thread.
    """

    def __init__(self, path, num_threads, getter_det_queue, det_pose_queue, frames):
        self.model = tf.compat.v1.lite.Interpreter(model_path=path, num_threads=num_threads)
        self.model.allocate_tensors()
        self.input_details = self.model.get_input_details()
        self.output_details = self.model.get_output_details()
        self.frame = None
        self.shape = 448
        self.humans = None
        self.stopped = True
        self.getter_det_queue = getter_det_queue
        self.det_pose_queue = det_pose_queue
        self.frames = frames

    def start(self):
        self.stopped = False
        Thread(target=self.detect, args=()).start()
        return self

    def detect(self):
        while not self.stopped:
            idx = self.getter_det_queue.get()
            if idx is None:
                for i in range(8):
                    self.det_pose_queue.put(None)

                self.stop()
                break
            # print("Detector", idx)
            self.frame = self.frames[idx].frame
            image = self.frame
            if image.shape[2] == 3:
                image = self.frame[:, :, [2, 1, 0]] # BGR to RGB

            if image.shape[2] == 1:
                image = tf.image.grayscale_to_rgb(image)
            input_image = cv2.resize(image, (self.shape, self.shape))

            input_image = np.expand_dims(input_image, axis=0)
            self.model.set_tensor(self.input_details[0]["index"], input_image)
            st = time.time()
            self.model.invoke()
            end = time.time()
            # print("detector", end - st)
            detection_boxes = self.model.get_tensor(self.output_details[0]["index"])
            detection_classes = self.model.get_tensor(self.output_details[1]["index"])
            detection_scores = self.model.get_tensor(self.output_details[2]["index"])
            num_detections = self.model.get_tensor(self.output_details[3]["index"])

            humans = []
            for i in range(int(num_detections[0])):
                detection_class = detection_classes[0, i]
                score = detection_scores[0, i]
                if detection_class == 0 and score > 0.3:
                    humans.append(detection_boxes[0, i])
            self.humans = np.asarray(humans) # Normalized [y1, x1, y2, x2]
            self.frames[idx].humans = self.humans
            self.det_pose_queue.put(idx)

    def stop(self):
        self.stopped = True
    
    def reset(self, getter_det_queue, det_pose_queue):
        self.getter_det_queue = getter_det_queue
        self.det_pose_queue = det_pose_queue
