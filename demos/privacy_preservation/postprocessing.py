from threading import Thread

import cv2
import numpy as np

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

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

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
            self.bboxes = self.frames[self.frame.detection_idx].humans
            self.blurred, self.pose = self.blur_humans(image)

            self.frames[idx].blurred = self.blurred
            self.frames[idx].pose = self.pose
            # cv2.imwrite(f"crops/{idx}.jpg", self.pose)
            # self.postprocessor_writter_queue.put(idx)
            self.postprocessor_writter_queue.append(idx)

            self.people = None
            self.frame_number += 1
            

    def stop(self):
        self.stopped = True
    
    def mask_face(self, mask, left_point, right_point, radius_multiplier):
        """
        Draws a circle on mask delimited by left_point and right_point.
        :param mask: nparray 
        :param left_point: array [x, y, score], either the left eye or left ear
        :param right_point: array [x, y, score], either the right eye or right ear
        :param radius_multiplier: float

        :return: ids: mask with the circle drawn on it
        """
        left_point = (int(left_point[1]), int(left_point[0]))
        right_point = (int(right_point[1]), int(right_point[0]))
        center = ((left_point[0] + right_point[0]) // 2, (left_point[1] + right_point[1]) // 2)
        radius = int(int_distance(left_point, center) * radius_multiplier)
        # cv2.circle(mask, center, radius, (255, 255, 255), cv2.FILLED)
        cv2.ellipse(mask, center, (int(radius * 1.5), radius), 90, 0, 360, (255, 255, 255), -1)

        return mask

    def blur_humans(self, npimg):
        npimg = npimg.astype("uint8")
        threshold = 0.11
        pose = npimg.copy()
        mask = np.zeros_like(npimg)

        img_blurred_bboxes = npimg.copy()
        centers = {}
        for idx, human in enumerate(self.people):
            bbox = [clamp(round(self.bboxes[idx][0] * npimg.shape[0]), 0, npimg.shape[0]),
                    clamp(round(self.bboxes[idx][1] * npimg.shape[1]), 0, npimg.shape[1]),
                    clamp(round(self.bboxes[idx][2] * npimg.shape[0]), 0, npimg.shape[0]),
                    clamp(round(self.bboxes[idx][3] * npimg.shape[1]), 0, npimg.shape[1])]
                    
            height = bbox[2] - bbox[0]
            width = bbox[3] - bbox[1]
            bboxed_img = npimg[bbox[0]:bbox[2], bbox[1]:bbox[3]]
            pixelated_size = (16, 1 + round(16 * (height/width)))
            temp = cv2.resize(bboxed_img, pixelated_size, interpolation=cv2.INTER_LINEAR)
            blur = cv2.resize(temp, (width, height), interpolation=cv2.INTER_NEAREST)
            img_blurred_bboxes[bbox[0]:bbox[2], bbox[1]:bbox[3]] = blur
            # cv2.rectangle(npimg, (bbox[1], bbox[0]), (bbox[3], bbox[2]), (255, 0, 0), 1) # Draw bboxes

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
            
            left_point, right_point, radius_multiplier = None, None, 1.0
            if human[0, KEYPOINT_DICT['left_ear'], 2] > threshold:
                left_point = human[0,  KEYPOINT_DICT['left_ear']]
            elif human[0, KEYPOINT_DICT['left_eye'], 2] > threshold:
                left_point = human[0, KEYPOINT_DICT['left_eye']]
                radius_multiplier *= 1.5
            
            if human[0, KEYPOINT_DICT['right_ear'], 2] > threshold:
                right_point = human[0, KEYPOINT_DICT['right_ear']]
            elif human[0, KEYPOINT_DICT['right_eye'], 2] > threshold:
                right_point = human[0, KEYPOINT_DICT['right_eye']]
                radius_multiplier *= 1.5

            if left_point is not None and right_point is None and human[0, KEYPOINT_DICT['nose'], 2] > threshold:
                right_point = human[0, KEYPOINT_DICT['nose']]
            if right_point is not None and left_point is None and human[0, KEYPOINT_DICT['nose'], 2] > threshold:
                left_point = human[0, KEYPOINT_DICT['nose']]
            
            if left_point is not None and right_point is not None:
                mask = self.mask_face(mask, left_point, right_point, radius_multiplier)
            
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
        npimg[mask>0] = img_blurred_bboxes[mask>0]
        
        return npimg, pose # npimg - blurred, pose - skeleton drawn over the image
