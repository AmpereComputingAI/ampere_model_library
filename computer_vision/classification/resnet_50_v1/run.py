# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import argparse

import numpy as np

from utils.benchmark import run_model
from utils.cv.imagenet import ImageNet
from utils.misc import print_goodbye_message_and_die, download_ampere_imagenet


def parse_args():
    parser = argparse.ArgumentParser(description="Run ResNet-50 v1 model.")
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32", "int8"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str,
                        choices=["tf", "tflite", "ort"], required=True,
                        help="specify the framework in which a model should be run")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--images_path",
                        type=str,
                        help="path to directory with ImageNet validation images")
    parser.add_argument("--labels_path",
                        type=str,
                        help="path to file with validation labels")
    parser.add_argument("--disable_jit_freeze", action='store_true',
                        help="if true model will be run not in jit freeze mode")
    args = parser.parse_args()
    if args.framework != "pytorch" and args.model_path is None:
        parser.error(f"You need to specify the model path when using {args.framework} framework.")
    return args


def run_tf_fp(model_path, batch_size, num_runs, timeout, images_path, labels_path):
    from utils.tf import TFSavedModelRunner
    import tensorflow as tf
    from tensorflow.python.saved_model import tag_constants

    def run_single_pass(tf_runner, imagenet):
        shape = (224, 224)
        output = tf_runner.run(batch_size, tf.constant(imagenet.get_input_array(shape)))
        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output['predictions'].numpy()[i]),
                imagenet.extract_top5(output['predictions'].numpy()[i])
            )

    dataset = ImageNet(batch_size, "BGR", images_path, labels_path,
                       pre_processing="VGG", is1001classes=False)
    runner = TFSavedModelRunner()
    saved_model_loaded = tf.saved_model.load(model_path, tags=[tag_constants.SERVING])
    runner.model = saved_model_loaded.signatures['serving_default']

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_tflite(model_path, batch_size, num_runs, timeout, images_path, labels_path):
    from utils.tflite import TFLiteRunner

    def run_single_pass(tflite_runner, imagenet):
        shape = (224, 224)
        tflite_runner.set_input_tensor(tflite_runner.input_details[0]['index'], imagenet.get_input_array(shape))
        tflite_runner.run(batch_size)
        output_tensor = tflite_runner.get_output_tensor(tflite_runner.output_details[0]['index'])
        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output_tensor[i]),
                imagenet.extract_top5(output_tensor[i])
            )

    dataset = ImageNet(batch_size, "BGR", images_path, labels_path,
                       pre_processing="VGG", is1001classes=False)
    runner = TFLiteRunner(model_path)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_ort_fp(model_path, batch_size, num_runs, timeout, images_path, labels_path):
    from utils.ort import OrtRunner

    def run_single_pass(ort_runner, imagenet):
        shape = (224, 224)
        ort_runner.set_input_tensor("input_tensor:0", imagenet.get_input_array(shape))
        output = ort_runner.run(batch_size)
        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output[1][i]),
                imagenet.extract_top5(output[1][i])
            )

    dataset = ImageNet(batch_size, "RGB", images_path, labels_path,
                       pre_processing="VGG", is1001classes=True, order="NCHW")
    runner = OrtRunner(model_path)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_tf_fp32(model_path, batch_size, num_runs, timeout, images_path, labels_path, **kwargs):
    return run_tf_fp(model_path, batch_size, num_runs, timeout, images_path, labels_path)


def run_tflite_int8(model_path, batch_size, num_runs, timeout, images_path, labels_path, **kwargs):
    return run_tflite(model_path, batch_size, num_runs, timeout, images_path, labels_path)


def run_ort_fp32(model_path, batch_size, num_runs, timeout, images_path, labels_path, **kwargs):
    return run_ort_fp(model_path, batch_size, num_runs, timeout, images_path, labels_path)


def main():
    args = parse_args()
    download_ampere_imagenet()

    if args.framework == "tf":
        if args.precision == "fp32":
            run_tf_fp32(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    elif args.framework == "tflite":
        if args.precision == "int8":
            run_tflite_int8(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    elif args.framework == "ort":
        if args.precision == "fp32":
            run_ort_fp32(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
