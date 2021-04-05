import tensorflow as tf


def load_graph(model):
    frozen_graph = model

    with tf.io.gfile.GFile(frozen_graph, "rb") as f:
        graph_def = tf.compat.v1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="")

    return graph



