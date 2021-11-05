import os
import sys
import torch
import argparse
import numpy as np

dlrm_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "dlrm")
sys.path.append(dlrm_path)
from dlrm.dlrm_s_pytorch import DLRM_Net
from utils.pytorch import PyTorchRunner
from utils.benchmark import run_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run DLRM model.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=True,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
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


def run_torch_fp32(model_path, batch_size, num_of_runs, timeout, images_path, labels_path):
    def run_single_pass(torch_runner, criteo):
        a, b, c = criteo.get_input()
        output = torch_runner.run(dense_x=a, lS_o=b, lS_i=c)
        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output["InceptionV3/Predictions/Reshape_1:0"][i]),
                imagenet.extract_top5(output["InceptionV3/Predictions/Reshape_1:0"][i])
            )

    # dataset = ImageNet(batch_size, "RGB", images_path, labels_path,
    #                    pre_processing="Inception", is1001classes=True)
    ln_top = np.array([479, 1024, 1024, 512, 256, 1])
    dlrm = DLRM_Net(
        m_spa=128,
        ln_emb=np.array(
            [39884406, 39043, 17289, 7420, 20263, 3, 7120, 1543, 63, 38532951, 2953546, 403346, 10, 2208, 11938, 155, 4,
             976, 14, 39979771, 25641295, 39664984, 585935, 12972, 108, 36]
        ),
        ln_bot=np.array([13, 512, 256, 128]),
        ln_top=ln_top,
        arch_interaction_op="dot",
        sigmoid_top=ln_top.size-2,
        # ndevices=self.ndevices,
        qr_operation=None,
        qr_collisions=None,
        qr_threshold=None,
        md_threshold=None,
    )
    dlrm.load_state_dict(torch.load(model_path)["state_dict"])
    runner = PyTorchRunner(dlrm)

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def main():
    args = parse_args()
    if args.precision == "fp32":
        run_torch_fp32(
            args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.labels_path
        )
    else:
        assert False, f"Behaviour undefined for precision {args.precision}"


if __name__ == "__main__":
    main()
