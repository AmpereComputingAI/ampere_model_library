# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
try:
    from utils import misc
except ModuleNotFoundError:
    import os
    import sys
    filename = "set_env_variables.sh"
    directory = os.path.realpath(__file__).split("/")[:-1]
    for idx in range(1, len(directory)-1):
        subdir = "/".join(directory[:-idx])
        if filename in os.listdir(subdir):
            print(f"\nPlease run \033[91m'source {os.path.join(subdir, filename)}'\033[0m first.")
            break
    else:
        print(f"\n\033[91mFAIL: Couldn't find {filename}, are you running this script as part of Ampere Model Library?"
              f"\033[0m")
    sys.exit(1)


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Run YOLOv5 model.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=True,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str,
                        choices=["pytorch", "ort"],
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
    import torch
    from utils.cv.coco import COCODataset
    from utils.benchmark import run_model
    from utils.cv.nms import non_max_suppression
    from utils.ort import OrtRunner

    def run_single_pass(ort_runner, coco):
        shape = (640, 640)
        ort_runner.set_input_tensor("images", coco.get_input_array(shape).astype("float32"))
        output = ort_runner.run(batch_size)

        output = torch.from_numpy(output[0])
        output = non_max_suppression(output)

        for i in range(batch_size):
            for d in range(output[i].shape[0]):
                coco.submit_bbox_prediction(
                    i,
                    coco.convert_bbox_to_coco_order(output[i][d][:4].tolist()),
                    output[i][d][4].item(),
                    coco.translate_cat_id_to_coco(output[i][d][5].item())
                )

    dataset = COCODataset(batch_size, "RGB", "COCO_val2014_000000000000", images_path,
                          anno_path, pre_processing="YOLO", order="NCHW")
    runner = OrtRunner(model_path)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch_fp(model_path, batch_size, num_runs, timeout, images_path, anno_path, disable_jit_freeze=False):
    import torch
    from utils.cv.coco import COCODataset
    from utils.benchmark import run_model
    from utils.cv.nms import non_max_suppression
    from utils.pytorch import PyTorchRunner

    def run_single_pass(pytorch_runner, coco):
        shape = (640, 640)
        inp = torch.stack(coco.get_input_array(shape))
        output = pytorch_runner.run(batch_size, inp)
        output = non_max_suppression(output)

        for i in range(batch_size):
            for d in range(output[i].shape[0]):
                coco.submit_bbox_prediction(
                    i,
                    coco.convert_bbox_to_coco_order(output[i][d][:4].tolist()),
                    output[i][d][4].item(),
                    coco.translate_cat_id_to_coco(int(output[i][d][5].item()))
                )

    dataset = COCODataset(batch_size, "RGB", "COCO_val2014_000000000000", images_path,
                          anno_path, pre_processing="PyTorch_objdet", sort_ascending=True, order="NCHW")

    from ultralytics import YOLO
    model = YOLO(model_path)
    torchscript_model = model.export(format="torchscript")

    runner = PyTorchRunner(torch.jit.load(torchscript_model),
                           disable_jit_freeze=disable_jit_freeze,
                           example_inputs=torch.stack(dataset.get_input_array((640, 640))))

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch_cuda(
        model_path, batch_size, num_runs, timeout, images_path, anno_path, disable_jit_freeze=False, **kwargs):
    import torch
    from utils.cv.coco import COCODataset
    from utils.benchmark import run_model
    from utils.pytorch import PyTorchRunnerV2

    def run_single_pass(pytorch_runner, coco):
        shape = (640, 640)
        inp = torch.tensor(coco.get_input_array(shape))
        output = pytorch_runner.run(batch_size, inp)

        for i in range(batch_size):
            for d in range(output[i].boxes.xywh.shape[0]):
                coco.submit_bbox_prediction(
                    i,
                    coco.convert_bbox_to_coco_order(output[i].boxes.xyxy[d, :].tolist()),
                    output[i].boxes.conf[d].item(),
                    coco.translate_cat_id_to_coco(output[i].boxes.cls[d].item())
                )

    dataset = COCODataset(batch_size, "RGB", "COCO_val2014_000000000000", images_path,
                          anno_path, pre_processing=None, sort_ascending=True, order="NCHW")

    from ultralytics import YOLO
    model = YOLO(model_path)

    runner = PyTorchRunnerV2(model)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch_fp32(model_path, batch_size, num_runs, timeout, images_path, anno_path, disable_jit_freeze, **kwargs):
    return run_pytorch_fp(model_path, batch_size, num_runs, timeout, images_path, anno_path, disable_jit_freeze)


def main():
    from utils.misc import print_goodbye_message_and_die
    args = parse_args()

    if args.framework == "pytorch":
        import torch
        if torch.cuda.is_available():
            run_pytorch_cuda(**vars(args))
        elif args.precision == "fp32":
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
