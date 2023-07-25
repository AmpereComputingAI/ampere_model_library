# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import argparse

import torch
import os

os.environ["YOLO_VERBOSE"] = os.getenv("YOLO_VERBOSE",
                                       "False")  # Ultralytics sets it to True by default. This way we suppress the logging by default while still allowing the user to set it to True if needed
from ultralytics.yolo.utils import ops

from utils.cv.coco import COCODataset
from utils.benchmark import run_model
from utils.misc import print_goodbye_message_and_die


def parse_args():
    parser = argparse.ArgumentParser(description="Run YOLOv8 model.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=True,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], default="fp32",
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str,
                        choices=["pytorch", "ort"], required=True,
                        help="specify the framework in which a model should be run")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--images_path",
                        type=str,
                        help="path to directory with COCO validation images")
    parser.add_argument("--anno_path",
                        type=str,
                        help="path to file with validation annotations")
    parser.add_argument("--disable_jit_freeze", action='store_true',
                        help="if true model will be run not in jit freeze mode")
    return parser.parse_args()


def run_ort_fp32(model_path, batch_size, num_runs, timeout, images_path, anno_path, **kwargs):
    from utils.ort import OrtRunner

    def run_single_pass(ort_runner, coco):
        shape = (640, 640)
        ort_runner.set_input_tensor("images", coco.get_input_array(shape).astype("float32"))
        output = ort_runner.run(batch_size)

        output = torch.from_numpy(output[0])
        output = ops.non_max_suppression(output)

        for i in range(batch_size):
            for d in range(output[i].shape[0]):
                coco.submit_bbox_prediction(
                    i,
                    coco.convert_bbox_to_coco_order(output[i][d][:4].tolist()),
                    output[i][d][4].item(),
                    coco.translate_cat_id_to_coco(output[i][d][5].item())
                )

    dataset = COCODataset(batch_size, "RGB", "COCO_val2014_000000000000", images_path, anno_path,
                          pre_processing="YOLO", order="NCHW")
    runner = OrtRunner(model_path)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch_fp(model_path, batch_size, num_runs, timeout, images_path, anno_path, disable_jit_freeze=False):
    from utils.pytorch import PyTorchRunner

    def run_single_pass(pytorch_runner, coco):
        shape = (640, 640)
        inp = torch.stack(coco.get_input_array(shape)).cuda()
        output = pytorch_runner.run(batch_size, inp)
        #print(output)
        #output = ops.non_max_suppression(output)

        for i in range(batch_size):
                coco.submit_bbox_prediction(
                    i,
                    coco.convert_bbox_to_coco_order([2, 0, 1, 2]),
                    1,
                    coco.translate_cat_id_to_coco(0)
                )

    dataset = COCODataset(batch_size, "RGB", "COCO_val2014_000000000000", images_path, anno_path,
                          pre_processing="PyTorch_objdet", sort_ascending=True, order="NCHW")

    import os
    import sys
    filename = "yolo_v8_m.ts"
    #model = torch.hub.load('ultralytics/yolov8', 'custom', 'yolov8m.engine')
    from ultralytics import YOLO
    model_name = YOLO(model_path).export(format="engine", device=0, half=True, imgsz=640, batch=batch_size)
    model = YOLO(f"{'/'.join(model_path.split('/')[:-1])}/{model_name}")
    #model = torch.jit.load(model).eval()
    #model = model.cuda()
    import os
    os.environ["BATCH_SIZE"] = str(batch_size)
    #import torch_tensorrt
    #if os.path.exists(filename):
    #    trt_ts_module = torch.jit.load(filename)
    #else:
    #    trt_ts_module = torch_tensorrt.compile(model, inputs=[torch_tensorrt.Input(min_shape=[1, 3, 640, 640], opt_shape=[128, 3, 640, 640], max_shape=[128, 3, 640, 640], dtype=torch.float)], enabled_precisions={torch.half})
    #    torch.jit.save(trt_ts_module, filename)
    #    print(model)
    #print(model)
    #print(torchscript_model)
    #device = torch.device("cuda:0")
    import numpy as np
    input_data = torch.from_numpy(np.random.rand(batch_size, 3, 640, 640).astype(np.float32))
    for _ in range(10):
        model(input_data, imgsz=640)#, batch=batch_size)
    import time
    latencies = []
    x = time.time()
    while time.time() - x < 60:
        start = time.time()
        model(input_data, imgsz=640)#, batch=batch_size)
        finish = time.time()
        latencies.append(finish-start)

    print(batch_size / (sum(latencies)/len(latencies)))
    #runner = PyTorchRunner(model,
    #                       disable_jit_freeze=disable_jit_freeze,
    #                       example_inputs=torch.stack(dataset.get_input_array((640, 640))).cuda())

    #return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch_fp32(model_path, batch_size, num_runs, timeout, images_path, anno_path, disable_jit_freeze, **kwargs):
    return run_pytorch_fp(model_path, batch_size, num_runs, timeout, images_path, anno_path, disable_jit_freeze)


def main():
    args = parse_args()

    if args.framework == "pytorch":
        if args.precision == "fp32":
            run_pytorch_fp32(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)
    elif args.framework == "ort":
        if args.precision == "fp32":
            if args.batch_size != 1:
                raise ValueError("Batch size must be 1 for this model.")
            run_ort_fp32(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)
    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
