import sys
import os.path
from os import path
import time, argparse
import tensorflow as tf
import numpy as np
import cv2
import math
from tensorflow.keras.preprocessing import image


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


# File responsible for RESNET
frozen_model_dir = "models/resnet50_v1.5_fp32.pb"


graph = load_graph(frozen_model_dir)
input_tensor = graph.get_tensor_by_name("input_tensor:0")
output_tensor = graph.get_tensor_by_name("softmax_tensor:0")
sess = tf.compat.v1.Session(graph=graph)

image_net = ImageNet()
preprocessed_image = image_net.get_input_tensor(32, (224,224), pre_process_vgg)

start = time.time()
result = sess.run(output_tensor, feed_dict={input_tensor: preprocessed_image})
end = time.time()
total_time += (end - start)


# file for graph loading

def load_graph(frozen_model_dir):
    frozen_graph = frozen_model_dir

    with tf.io.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph


if __name__ == "__main__":
    main()
