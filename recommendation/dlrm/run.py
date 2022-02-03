import os
import sys
import torch
import argparse
import numpy as np

from utils.recommendation.criteo import Criteo, append_dlrm_to_pypath
from utils.pytorch import PyTorchRunner
from utils.benchmark import run_model

from utils.misc import UnsupportedPrecisionValueError, FrameworkUnsupportedError


def parse_args():
    parser = argparse.ArgumentParser(description="Run DLRM model.")
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=2048,
                        help="batch size to feed the model with")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--dataset_path",
                        type=str,
                        help="path to Criteo dataset .txt file")
    parser.add_argument("--framework",
                        type=str,
                        choices=["pytorch"], required=True,
                        help="specify the framework in which a model should be run")
    return parser.parse_args()


def run_torch_fp(model_path, batch_size, num_runs, timeout, dataset_path, **kwargs):

    def run_single_pass(torch_runner, criteo):
        _ = torch_runner.run(criteo.get_inputs())

    append_dlrm_to_pypath()
    from utils.recommendation.dlrm.dlrm_s_pytorch import DLRM_Net

    dataset = Criteo(max_batch_size=batch_size, dataset_path=dataset_path)

    ln_top = np.array([479, 1024, 1024, 512, 256, 1])
    dlrm = DLRM_Net(
        m_spa=128,
        ln_emb=dataset.ln_emb,
        ln_bot=np.array([13, 512, 256, 128]),
        ln_top=ln_top,
        arch_interaction_op="dot",
        sigmoid_top=ln_top.size-2,
        qr_operation="mult",
        qr_collisions=4,
        qr_threshold=200,
        md_threshold=200,
    )
    dlrm.load_state_dict(torch.load(model_path)["state_dict"])

    runner = PyTorchRunner(dlrm)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch_fp32(**kwargs):
    return run_pytorch_fp(**kwargs)


def main():
    args = parse_args()
    if args.framework == "pytorch":
        if args.model_path is None:
            print_goodbye_message_and_die(
                "a path to model is unspecified!")

        if args.precision == "fp32":
            run_torch_fp32(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
