import sys
import os.path
from os import path
import time
import tensorflow as tf
import numpy as np
import cv2
import math
from tensorflow.keras.preprocessing import image



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
