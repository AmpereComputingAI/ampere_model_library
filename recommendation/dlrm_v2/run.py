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
    parser = argparse.ArgumentParser(description="Run DLRMv2 model.")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=2048,
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
    return parser.parse_args()


def run_pytorch_fp(batch_size, num_runs, timeout):
    import torch
    from torchrec import EmbeddingBagCollection
    from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
    from torchrec.models.dlrm import DLRM_DCN
    from torchrec.modules.embedding_configs import EmbeddingBagConfig


    from utils.benchmark import run_model
    from utils.pytorch import PyTorchRunnerV2, apply_compile
    from utils.recommendation.dlrm_v2_random import RandomDataset

    def run_single_pass(torch_runner, dataset):
        output = torch_runner.run(batch_size, *dataset.get_inputs())
        dataset.submit_predictions(output)

    device = "cpu"

    embedding_dim = 128
    num_embeddings_per_feature = [40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36]
    eb_configs = [
                EmbeddingBagConfig(
                    name=f"t_{feature_name}",
                    embedding_dim=embedding_dim,
                    num_embeddings=num_embeddings_per_feature[feature_idx],
                    feature_names=[feature_name],
                )
                for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
            ]

    dense_arch_layer_sizes = [512, 256, 128]
    over_arch_layer_sizes = [1024, 1024, 512, 256, 1]
    dcn_num_layers = 3
    dcn_low_rank_dim = 512

    dlrm_model = DLRM_DCN(
                embedding_bag_collection=EmbeddingBagCollection(
                    tables=eb_configs, device=torch.device(device)
                ),
                dense_in_features=len(DEFAULT_INT_NAMES),
                dense_arch_layer_sizes=dense_arch_layer_sizes,
                over_arch_layer_sizes=over_arch_layer_sizes,
                dcn_num_layers=dcn_num_layers,
                dcn_low_rank_dim=dcn_low_rank_dim,
                dense_device=device,
            )

    dlrm_model = apply_compile(dlrm_model)


    runner = PyTorchRunnerV2(dlrm_model)
    dataset = RandomDataset(batch_size)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch_fp32(batch_size, num_runs, timeout, **kwargs):
    return run_pytorch_fp(batch_size, num_runs, timeout)


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
