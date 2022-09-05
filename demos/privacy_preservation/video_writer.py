from pathlib import Path
import subprocess
import sys
import time
from threading import Thread

class VideoWriter:
    """
    Class that continuously writes frames to file using a dedicated thread.
    """

    def __init__(self, queue, frames, save):
        self.frame = None
        self.stopped = False
        self.queue = queue
        self.frames = frames
        self.save = save
        self.frame_number = 0
        self.last_deleted_idx = 0
        self.last_frame = -1
        self.proc = None

    def start(self, source, fps, height, width, num_frames):
        out_path = f"out/{Path(str(source)).stem}.avi"
        self.command = ["ffmpeg",
                        "-y", # overwrite
                        "-f", "rawvideo", "-vcodec", "rawvideo",
                        "-s", f"{int(width)}x{int(height)}",
                        "-pix_fmt", "rgb24", "-r", f"{fps}",
                        "-i", "-", # input comes from pipe
                        "-an", # no audio
                        "-vcodec", "libx264", "-preset", "ultrafast", out_path] # mpeg4 - quality is terrible, ffv1 - file size is huge
        if self.save:
            self.proc = subprocess.Popen(self.command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
        else:
            self.proc = None
        self.last_frame = int(num_frames) - 2 if num_frames > 0 else sys.maxsize
        self.stopped = False
        Thread(target=self.write, args=()).start()
        return self

    def write(self):
        while not self.stopped:
            if self.frame_number > self.last_frame:
                self.stop()
                break
            while self.frame_number not in self.queue:
                time.sleep(0.001)
            idx = self.frame_number
            if idx is None:
                self.stop()
                break
            # print("Writer ", idx)
            self.frame = self.frames[idx].blurred
            # self.frame = self.frames[idx].pose
            # self.frame = self.frames[idx].frame
            # self.frame = np.concatenate((self.frames[idx].frame, self.frames[idx].blurred), axis=1)

            if self.save:
                self.frame = self.frame[:, :, [2, 1, 0]] # RGB to BGR
                try:
                    self.proc.stdin.write(self.frame.tostring())
                except ValueError:
                    pass
            self.frame_number += 1


    def stop(self):
        self.stopped = True
        if self.proc:
            self.proc.stdin.close()
            if self.proc.stderr is not None:
                self.proc.stderr.close()
            self.proc.wait()
    
    def reset(self, queue):
        self.queue = queue
        self.frame_number = 0
        self.last_deleted_idx = 0
