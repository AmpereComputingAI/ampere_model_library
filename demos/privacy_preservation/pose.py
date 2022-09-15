import concurrent.futures
from threading import Thread, current_thread
import time
import numpy as np
import tensorflow as tf
import queue


class Pose:
    """
    Class that continuously detects objects on image using a dedicated thread.
    """

    def __init__(self, path, num_threads, det_pose_queue, pose_postprocessor_queue, frames):
        if "lightning" in path:
            self.shape = 192
        else:
            self.shape = 256
        self.frame = None # Full frame
        self.humans = None
        self.stopped = True
        self.det_pose_queue = det_pose_queue
        self.pose_postprocessor_queue = pose_postprocessor_queue
        self.frames = frames
        self.interpreters = {}
        self.executor = concurrent.futures.ThreadPoolExecutor(max_workers=10, initializer=self.init_threads, initargs=(self.interpreters, path, num_threads))
    
    def init_threads(self, interpreters, model_path, num_threads):
        model = tf.compat.v1.lite.Interpreter(model_path=model_path, num_threads=num_threads)
        model.allocate_tensors()
        interpreters[current_thread().name] = model

    def run(self, interpreters, image, idx):
        model = interpreters[current_thread().name]
        model.set_tensor(model.get_input_details()[0]['index'], image)
        st = time.time()
        model.invoke()
        end = time.time()
        # print("pose", end - st)
        output_tensor = model.get_tensor(model.get_output_details()[0]['index'])
        return output_tensor, idx

    def start(self):
        self.stopped = False
        Thread(target=self.detect, args=()).start()
        return self

    def detect(self):
        while not self.stopped:
            try:
                idx = self.det_pose_queue.get(block=False)
            except queue.Empty:
                time.sleep(0.001)
                continue
            if idx is None:
                for i in range(8):
                    self.pose_postprocessor_queue.append(None)

                self.stop()
                break
            # print("Pose", idx)
            self.frame = self.frames[idx].frame
            self.frame = self.frame[:, :, [2, 1, 0]] # BGR to RGB

            self.humans = self.frames[idx].humans
            if self.humans.shape[0] != 0:
                input_image = np.expand_dims(self.frame, axis=0)
                input_image = tf.cast(tf.image.crop_and_resize(input_image, self.humans, [0]*self.humans.shape[0], (self.shape, self.shape)), dtype=tf.float32)
            keypoints_with_scores = []
            futures = [self.executor.submit(self.run, self.interpreters, input_image[i:i+1].numpy(), i) for i in range(len(self.humans))]
            for future in concurrent.futures.as_completed(futures):
                pose, i = future.result()
                human = self.humans[i]
                cropped_height = int((human[2] - human[0]) * self.frame.shape[0])
                cropped_width = int((human[3] - human[1]) * self.frame.shape[1])
                offset_height = int(human[0] * self.frame.shape[0])
                offset_width = int(human[1] * self.frame.shape[1])

                pose[0, 0, :, 0] = pose[0, 0, :, 0] * cropped_height + offset_height
                pose[0, 0, :, 1] = pose[0, 0, :, 1] * cropped_width + offset_width

                keypoints_with_scores.append(pose)

            self.frames[idx].people = np.asarray(keypoints_with_scores)
            self.pose_postprocessor_queue.append(idx)

            self.humans = None

    def stop(self):
        self.stopped = True
    
    def reset(self, det_pose_queue, pose_postprocessor_queue):
        self.det_pose_queue = det_pose_queue
        self.pose_postprocessor_queue = pose_postprocessor_queue
        self.humans = None
