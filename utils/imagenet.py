import sys
import os.path
from os import path
import math
import numpy as np
import cv2
from utils.mix import batch

labels = 'model_zoo/utils/ILSVRC2012_validation_ground_truth.txt'


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
        final_batch = np.empty((0, 224, 224, 3))

        parent_list = os.listdir(self.images_path)
        g = batch(parent_list, batch_size)

        for i in g.__next__():

            img = cv2.imread(os.path.join(self.images_path, i))
            resized_img = cv2.resize(img, input_shape)
            preprocessed_img = preprocess(resized_img)
            final_batch = np.append(final_batch, preprocessed_img, axis=0)

        return final_batch

