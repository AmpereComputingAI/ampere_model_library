# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import argparse

from utils.benchmark import run_model
from utils.cv.imagenet import ImageNet
from utils.misc import print_goodbye_message_and_die, download_ampere_imagenet


def parse_args():
    parser = argparse.ArgumentParser(description="Run Inception v2 model.")
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32", "fp16", "int8"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str,
                        choices=["tf", "ort"], required=True,
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
    return parser.parse_args()


def run_tf_fp(model_path, batch_size, num_runs, timeout, images_path, labels_path):
    from utils.tf import TFFrozenModelRunner

    def run_single_pass(tf_runner, imagenet):
        shape = (224, 224, 3)
        tf_runner.set_input_tensor("input:0", imagenet.get_input_array(shape))
        output = tf_runner.run()
        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output["InceptionV2/Predictions/Reshape_1:0"][i]),
                imagenet.extract_top5(output["InceptionV2/Predictions/Reshape_1:0"][i])
            )

    dataset = ImageNet(batch_size, "RGB", images_path, labels_path,
                       pre_processing="Inception", is1001classes=True)
    runner = TFFrozenModelRunner(model_path, ["InceptionV2/Predictions/Reshape_1:0"])

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_tflite(model_path, batch_size, num_runs, timeout, images_path, labels_path):
    from utils.tflite import TFLiteRunner

    def run_single_pass(tflite_runner, imagenet):
        shape = (224, 224)
        tflite_runner.set_input_tensor(tflite_runner.input_details[0]['index'], imagenet.get_input_array(shape))
        tflite_runner.run()
        output_tensor = tflite_runner.get_output_tensor(tflite_runner.output_details[0]['index'])
        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output_tensor[i]),
                imagenet.extract_top5(output_tensor[i])
            )

    dataset = ImageNet(batch_size, "RGB", images_path, labels_path,
                       pre_processing="Inception", is1001classes=True)
    runner = TFLiteRunner(model_path)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)
  

def run_ort_fp(model_path, batch_size, num_runs, timeout, images_path, labels_path):
    from utils.ort import OrtRunner

    def run_single_pass(ort_runner, imagenet):
        shape = (224, 224)
        ort_runner.set_input_tensor("input:0", imagenet.get_input_array(shape).astype("float16"))
        output = ort_runner.run()[0]
        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output[i]),
                imagenet.extract_top5(output[i])
            )

    dataset = ImageNet(batch_size, "RGB", images_path, labels_path,
                       pre_processing="Inception", is1001classes=True, order="NHWC")
    runner = OrtRunner(model_path)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_tf_fp32(model_path, batch_size, num_runs, timeout, images_path, labels_path, **kwargs):
    return run_tf_fp(model_path, batch_size, num_runs, timeout, images_path, labels_path)


def run_tf_fp16(model_path, batch_size, num_runs, timeout, images_path, labels_path, **kwargs):
    return run_tf_fp(model_path, batch_size, num_runs, timeout, images_path, labels_path)


def run_tflite_int8(model_path, batch_size, num_runs, timeout, images_path, labels_path, **kwargs):
    return run_tflite(model_path, batch_size, num_runs, timeout, images_path, labels_path)
  

def run_ort_fp16(model_path, batch_size, num_runs, timeout, images_path, labels_path, **kwargs):
    return run_ort_fp(model_path, batch_size, num_runs, timeout, images_path, labels_path)


def main():
    args = parse_args()
    download_ampere_imagenet()

    if args.framework == "tf":
        if args.model_path is None:
            print_goodbye_message_and_die(
                "a path to model is unspecified!")

        if args.precision == "fp32":
            run_tf_fp32(**vars(args))
        elif args.precision == "fp16":
            run_tf_fp16(**vars(args))
        elif args.precision == "int8":
            run_tflite_int8(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    elif args.framework == "ort":
        if args.model_path is None:
            print_goodbye_message_and_die(
                "a path to model is unspecified!")
            
        if args.precision == "fp16":
            run_ort_fp16(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
