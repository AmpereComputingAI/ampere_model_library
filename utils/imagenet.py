import sys
import os.path
from os import path
import math
import numpy as np
import cv2
from utils.mix import batch


class ImageNet:

    # variable which you can set by initializing ImageNet

    def __init__(self):

        self.images_path = os.environ['IMAGES_PATH']
        self.number_of_images = 50000
        self.image_count = 0

        if not path.exists(self.images_path):
            print("path doesn't exist")
            sys.exit(1)
        else:
            print('works')

    def get_input_tensor(self, batch_size, input_shape, preprocess):
        final_image = np.empty(0)

        print(final_image)
        print(final_image.shape)
        print(self.number_of_images / batch_size)

        parent_list = os.listdir(self.images_path)

        g = batch(parent_list, 4)

        for x in g.__next__():
            print(x)
