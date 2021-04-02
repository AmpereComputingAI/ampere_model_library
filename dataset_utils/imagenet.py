import sys
import os.path
from os import path
import cv2
import math
import numpy as np
import cv2

class ImageNet:

    # zmienna którą da się ustawić inicjalizując imageNet

    # self.images_path = 'images/ILSVRC2012_img_val' <- wrzucić w zmienną środowiskową os.environ

    # yield zamiast 'return'

    def __init__(self):

        self.images_path = 'images/ILSVRC2012_img_val'
        self.image = 'ILSVRC2012_val_00000000'
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

        while self.image_count < self.number_of_images:
            for image_n in range(1, batch_size):
                self.image_count += 1

                digits = int(math.log10(image_n)) + 1

                path_to_image = os.path.join(self.images_path, self.image[:-digits] + str(image_n) + '.JPEG')

                img = cv2.imread(path_to_image)
                resized_image_n = cv2.resize(img, input_shape)
                print(resized_image_n.shape)
                image_n = preprocess(resized_image_n)
                print(image_n.shape)
                final_image.append(image_n)

                print(type(final_image))

ImageNet()