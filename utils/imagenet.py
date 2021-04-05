import sys
import os.path
from os import path
import math
import numpy as np
import cv2
from utils.mix import batch


class ImageNet:

    # variable which you can set by initializing ImageNet

    def __init__(self, batch_size):

        self.images_path = os.environ['IMAGES_PATH']
        self.number_of_images = 50000
        self.image_count = 0
        self.parent_list = os.listdir(self.images_path)
        self.g = batch(self.parent_list, batch_size)

        if not path.exists(self.images_path):
            print("path doesn't exist")
            sys.exit(1)
        else:
            print('works')

    def get_input_tensor(self, input_shape, preprocess):
        final_batch = np.empty((0, 224, 224, 3))

        for i in self.g.__next__():

            img_path = os.path.join(self.images_path, i)
            img = cv2.imread(os.path.join(self.images_path, i))
            resized_img = cv2.resize(img, input_shape)
            preprocessed_img = preprocess(resized_img)
            final_batch = np.append(final_batch, preprocessed_img, axis=0)

        return img_path, final_batch

