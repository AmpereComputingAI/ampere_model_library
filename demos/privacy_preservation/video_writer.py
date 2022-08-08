import subprocess
import sys
import time
from threading import Thread

import cv2


class VideoWriter:
    """
    Class that continuously writes frames to file using a dedicated thread.
    """

    def __init__(self, out_path, fps, width, height, queue, frames, show, num_frames):
        self.frame = None
        self.stopped = False
        self.command = ["ffmpeg",
                        "-y", # overwrite
                        "-f", "rawvideo", "-vcodec", "rawvideo",
                        "-s", f"{int(width)}x{int(height)}",
                        "-pix_fmt", "rgb24", "-r", f"{fps}",
                        "-i", "-", # input comes from pipe
                        "-an", # no audio
                        "-vcodec", "libx264", "-preset", "ultrafast", out_path] # mpeg4 - quality is terrible, ffv1 - file size is huge
        self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        self.queue = queue
        self.frames = frames
        self.show = show
        self.frame_number = 0
        self.last_frame = int(num_frames) - 2 if num_frames > 0 else sys.maxsize
        self.last_deleted_idx = 0

    def start(self):
        Thread(target=self.write, args=()).start()
        return self

    def write(self):
        while not self.stopped:
            while self.frame_number not in self.queue:
                if self.frame_number > self.last_frame:
                    self.stop()
                    break # Break out of the inner loop
                time.sleep(0.001)
            if self.frame_number > self.last_frame: 
                break # Break out of the outer loop
            idx = self.frame_number
            if idx is None:
                self.stop()
                break
            # print("Writer ", idx)
            self.frame = self.frames[idx].blurred
            # self.frame = self.frames[idx].pose
            # self.frame = self.frames[idx].frame
            # self.frame = np.concatenate((self.frames[idx].frame, self.frames[idx].blurred), axis=1)

            if self.show:
                cv2.imshow("Privacy Preservation Demo", self.frame)
                cv2.waitKey(1)
            self.frame = self.frame[:, :, [2, 1, 0]] # RGB to BGR
            self.proc.stdin.write(self.frame.tostring())
            self.frame_number += 1
            if self.frames[idx].detection_idx == idx:
                for i in range(self.last_deleted_idx, idx):
                    self.frames[i] = None
                    self.last_deleted_idx = idx

            # self.frames.pop(idx)


    def stop(self):
        self.stopped = True
        if self.proc:
            self.proc.stdin.close()
            if self.proc.stderr is not None:
                self.proc.stderr.close()
            self.proc.wait()
