# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
try:
    from utils import misc  # noqa
except ModuleNotFoundError:
    import os
    import sys
    filename = "set_env_variables.sh"
    directory = os.path.realpath(__file__).split("/")[:-1]
    for idx in range(1, len(directory) - 1):
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
    parser = argparse.ArgumentParser(description="Run SSD VGG-16 model.")
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str, default="pytorch",
                        choices=["pytorch"],
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


def run_pytorch_fp(batch_size, num_runs, timeout, images_path, anno_path, disable_jit_freeze=False):
    import os
    import torchvision
    from utils.cv.coco import COCODataset
    from utils.benchmark import run_model
    from utils.pytorch import PyTorchRunner

    def run_single_pass(pytorch_runner, coco):
        shape = (300, 300)
        output = pytorch_runner.run(batch_size, coco.get_input_array(shape))
        if not disable_jit_freeze and not os.environ.get("TORCH_COMPILE") == "1":
            output = output[1]

        for i in range(batch_size):
            for d in range(output[i]['boxes'].shape[0]):
                coco.submit_bbox_prediction(
                    i,
                    coco.convert_bbox_to_coco_order(output[i]['boxes'][d].tolist()),
                    output[i]['scores'][d].item(),
                    output[i]['labels'][d].item()
                )

    dataset = COCODataset(batch_size, "BGR", "COCO_val2014_000000000000", images_path,
                          anno_path, pre_processing="PyTorch_objdet", sort_ascending=True, order="NCHW")
    runner = PyTorchRunner(torchvision.models.detection.ssd300_vgg16(pretrained=True),
                           disable_jit_freeze=disable_jit_freeze)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch_fp32(batch_size, num_runs, timeout, images_path, anno_path, disable_jit_freeze, **kwargs):
    return run_pytorch_fp(batch_size, num_runs, timeout, images_path, anno_path, disable_jit_freeze)


def main():
    from utils.misc import print_goodbye_message_and_die
    args = parse_args()
    if args.framework == "pytorch":
        if args.precision == "fp32":
            run_pytorch_fp32(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
