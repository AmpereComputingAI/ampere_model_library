from threading import Thread
import numpy as np
import cv2
import time
# Dictionary that maps from joint names to keypoint indices.
KEYPOINT_DICT = {
    'nose': 0,
    'left_eye': 1,
    'right_eye': 2,
    'left_ear': 3,
    'right_ear': 4,
    'left_shoulder': 5,
    'right_shoulder': 6,
    'left_elbow': 7,
    'right_elbow': 8,
    'left_wrist': 9,
    'right_wrist': 10,
    'left_hip': 11,
    'right_hip': 12,
    'left_knee': 13,
    'right_knee': 14,
    'left_ankle': 15,
    'right_ankle': 16
}

# Maps bones to a matplotlib color name.
KEYPOINT_EDGE_INDS_TO_COLOR = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

def int_distance(point1, point2):
    return int(np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2))


class Postprocessor:
    """
    Class that continuously blurs people on image using a dedicated thread.
    """

    def __init__(self, pose_postprocessor_queue, postprocessor_writter_queue, frames, faces):
        self.frame = None
        self.people = None
        self.blurred = None
        self.pose = None
        self.stopped = False
        self.pose_postprocessor_queue = pose_postprocessor_queue
        self.postprocessor_writter_queue = postprocessor_writter_queue
        self.frames = frames
        self.frame_number = 0
        self.faces = faces

    def start(self):
        Thread(target=self.blur, args=()).start()
        return self

    def blur(self):
        while not self.stopped:
            # idx = self.pose_postprocessor_queue.get()
            while self.frame_number not in self.pose_postprocessor_queue:
                if None in self.pose_postprocessor_queue and self.frame_number > max(el for el in self.pose_postprocessor_queue if el is not None):
                    self.stop()
                    break # Break out of the inner loop
                continue
            if self.frame_number > max(el for el in self.pose_postprocessor_queue if el is not None): 
                break # Break out of the outer loop
            idx = self.frame_number
            if idx is None:
                self.stop()
                break
            # print("Post", idx)
            self.frame = self.frames[idx]
            image = self.frame.frame
            self.people = self.frames[self.frame.detection_idx].people
            self.blurred, self.pose = self.blur_humans(image, self.people)

            self.frames[idx].blurred = self.blurred
            self.frames[idx].pose = self.pose
            # cv2.imwrite(f"crops/{idx}.jpg", self.pose)
            # self.postprocessor_writter_queue.put(idx)
            self.postprocessor_writter_queue.append(idx)

            self.people = None
            self.frame_number += 1
            

    def stop(self):
        self.stopped = True
    
    def mask_face(self, mask, left_point, right_point):
        """
        Draws a circle on mask delimited by left_point and right_point.
        :param mask: nparray 
        :param left_point: array [x, y, score], either the left eye or left ear
        :param right_point: array [x, y, score], either the right eye or right ear 

        :return: ids: mask with the circle drawn on it
        """
        left_point = (int(left_point[1]), int(left_point[0]))
        right_point = (int(right_point[1]), int(right_point[0]))
        center = ((left_point[0] + right_point[0]) // 2, (left_point[1] + right_point[1]) // 2)
        radius = int_distance(left_point, center) * 2
        cv2.circle(mask, center, radius, (255, 255, 255), cv2.FILLED)

        return mask

    def blur_humans(self, npimg, humans):
        npimg = npimg.astype("uint8")
        threshold = 0.11
        pose = npimg.copy()
        mask = np.zeros_like(npimg)

        height, width = npimg.shape[:2]
        pixelated_size = int(width/10)
        # Resize input to "pixelated" size
        temp = cv2.resize(npimg, (pixelated_size, pixelated_size), interpolation=cv2.INTER_LINEAR)
        blur = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)

        image_h, image_w = mask.shape[:2]
        centers = {}
        for human in humans:
            human = human[0]
            # draw point
            for key, value in KEYPOINT_DICT.items():
                if human[0, value, 2] < threshold:
                    continue

                body_part = human[0, value]
                center = (int(body_part[1]), int(body_part[0]))
                centers[value] = center
                cv2.circle(mask, center, 3, [128, 0, 128], thickness=3, lineType=8, shift=0)

            # draw line
            lines = []
            for pair in KEYPOINT_EDGE_INDS_TO_COLOR:
                if human[0, pair[0], 2] < threshold or human[0, pair[1], 2] < threshold:
                    continue
                lines.append([centers[pair[0]], centers[pair[1]]])
            
            left_point, right_point = None, None
            if human[0, KEYPOINT_DICT['left_ear'], 2] > threshold:
                left_point = human[0,  KEYPOINT_DICT['left_ear']]
            elif human[0, KEYPOINT_DICT['left_eye'], 2] > threshold:
                left_point = human[0, KEYPOINT_DICT['left_eye']]
            
            if human[0, KEYPOINT_DICT['right_ear'], 2] > threshold:
                right_point = human[0, KEYPOINT_DICT['right_ear']]
            elif human[0, KEYPOINT_DICT['right_eye'], 2] > threshold:
                right_point = human[0, KEYPOINT_DICT['right_eye']]
            
            if left_point is not None and right_point is not None:
                mask = self.mask_face(mask, left_point, right_point)
            
            if not self.faces:
                # Blur torso
                if (human[0, KEYPOINT_DICT['left_shoulder'], 2] > threshold and
                    human[0, KEYPOINT_DICT['right_shoulder'], 2] > threshold and
                    human[0, KEYPOINT_DICT['left_hip'], 2] > threshold and
                    human[0, KEYPOINT_DICT['right_hip'], 2] > threshold):
                    contours = np.array([
                        human[0, KEYPOINT_DICT['left_shoulder'], :2],
                        human[0, KEYPOINT_DICT['right_shoulder'], :2],
                        human[0, KEYPOINT_DICT['right_hip'], :2],
                        human[0, KEYPOINT_DICT['left_hip'], :2],
                    ], dtype="int64")
                    contours = contours[:, [1, 0]]
                    cv2.fillPoly(mask, pts=[contours], color=(255, 255, 255))
                lines = np.array(lines, dtype=np.uint64)

                # Determine the thickness of the lines
                thickness = 10
                if (human[0, KEYPOINT_DICT['left_shoulder'], 2] > threshold and
                    human[0, KEYPOINT_DICT['left_elbow'], 2] > threshold):
                    point1 = human[0, KEYPOINT_DICT['left_shoulder'], :2]
                    point2 = human[0, KEYPOINT_DICT['left_elbow'], :2]
                    thickness = int_distance(point1, point2) // 3
                elif (human[0, KEYPOINT_DICT['right_shoulder'], 2] > threshold and
                    human[0, KEYPOINT_DICT['right_elbow'], 2] > threshold):
                    point1 = human[0, KEYPOINT_DICT['right_shoulder'], :2]
                    point2 = human[0, KEYPOINT_DICT['right_elbow'], :2]
                    thickness = int_distance(point1, point2) // 3
                elif (human[0, KEYPOINT_DICT['left_wrist'], 2] > threshold and
                    human[0, KEYPOINT_DICT['left_elbow'], 2] > threshold):
                    point1 = human[0, KEYPOINT_DICT['left_wrist'], :2]
                    point2 = human[0, KEYPOINT_DICT['left_elbow'], :2]
                    thickness = int_distance(point1, point2) // 3
                elif (human[0, KEYPOINT_DICT['right_wrist'], 2] > threshold and
                    human[0, KEYPOINT_DICT['right_elbow'], 2] > threshold):
                    point1 = human[0, KEYPOINT_DICT['right_wrist'], :2]
                    point2 = human[0, KEYPOINT_DICT['right_elbow'], :2]
                    thickness = int_distance(point1, point2) // 3
                elif (human[0, KEYPOINT_DICT['left_shoulder'], 2] > threshold and
                    human[0, KEYPOINT_DICT['right_shoulder'], 2] > threshold):
                    point1 = human[0, KEYPOINT_DICT['left_shoulder'], :2]
                    point2 = human[0, KEYPOINT_DICT['right_shoulder'], :2]
                    thickness = int_distance(point1, point2) // 3
                elif (human[0, KEYPOINT_DICT['left_hip'], 2] > threshold and
                    human[0, KEYPOINT_DICT['right_hip'], 2] > threshold):
                    point1 = human[0, KEYPOINT_DICT['left_hip'], :2]
                    point2 = human[0, KEYPOINT_DICT['right_hip'], :2]
                    thickness = int_distance(point1, point2) // 3
                cv2.polylines(mask, lines, False, (255, 255, 255), thickness)
        pose[mask>0] = mask[mask>0]
        npimg[mask>0] = blur[mask>0]
        
        return npimg, pose # npimg - blurred, pose - skeleton drawn over the image