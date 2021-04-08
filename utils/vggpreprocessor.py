import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


def vgg_preprocessor(image_sample, model):
    img_array = image.img_to_array(image_sample)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    if model == 'resnet':
        result = tf.keras.applications.resnet.preprocess_input(img_array_expanded_dims)
    elif model == 'mobilenet':
        result = tf.keras.applications.mobilenet.preprocess_input(img_array_expanded_dims)

    return result
