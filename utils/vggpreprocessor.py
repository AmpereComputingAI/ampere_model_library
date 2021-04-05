import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


def vgg_preprocessor(image_sample):

    img_array = image.img_to_array(image_sample)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    result = tf.keras.applications.resnet.preprocess_input(img_array_expanded_dims)

    return result
