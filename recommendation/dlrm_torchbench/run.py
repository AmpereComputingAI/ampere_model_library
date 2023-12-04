# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import argparse

import numpy as np

from utils.benchmark import run_model
from utils.misc import print_goodbye_message_and_die
from utils.recommendation.criteo import append_dlrm_to_pypath
from utils.recommendation.torchbench_random import RandomDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run torch/benchmark DLRM model.")
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
    from utils.pytorch import PyTorchRunner

    def run_single_pass(torch_runner, dataset):
        output = torch_runner.run(batch_size, *example_inputs)
        dataset.submit_predictions(output)

    append_dlrm_to_pypath()
    from utils.recommendation.dlrm.dlrm_s_pytorch import DLRM_Net

    arch_embedding_size = "1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000"
    arch_sparse_feature_size = 64
    arch_mlp_bot = "512-512-64"
    arch_mlp_top = "1024-1024-1024-1"
    data_generation = "random"
    mini_batch_size = batch_size
    num_batches = 1
    num_indicies_per_lookup = 100

    opt = argparse.Namespace(**{
        'm_spa': None,
        'ln_emb': None,
        'ln_bot': None,
        'ln_top': None,
        'arch_interaction_op': "dot",
        'arch_interaction_itself': False,
        'sigmoid_bot': -1,
        'sigmoid_top': -1,
        'sync_dense_params': True,
        'ndevices': -1,
        'qr_flag': False,
        'qr_operation': "mult",
        'qr_collisions': 0,
        'qr_threshold': 200,
        'md_flag': False,
        'md_threshold': 200,
        'md_temperature': 0.3,
        'activation_function': "relu",
        'loss_function': "bce",
        'loss_weights': "1.0-1.0",
        'loss_threshold': 0.0,
        'round_targets': False,
        'data_size': 6,
        'data_generation': data_generation,
        'data_trace_file': "./input/dist_emb_j.log",
        'raw_data_file': "",
        'processed_data_file': "",
        'data_randomize': "total",
        'data_trace_enable_padding': False,
        'max_ind_range': -1,
        'num_workers': 0,
        'memory_map': False,
        'data_sub_sample_rate': 0.0,
        'learning_rate': 0.01,
        'lr_num_warmup_steps': 0,
        'lr_decay_start_step': 0,
        'lr_num_decay_steps': 0,
        'arch_embedding_size': arch_embedding_size,
        'arch_sparse_feature_size': arch_sparse_feature_size,
        'arch_mlp_bot': arch_mlp_bot,
        'arch_mlp_top': arch_mlp_top,
        'mini_batch_size': mini_batch_size,
        'num_batches': num_batches,
        'num_indices_per_lookup': num_indicies_per_lookup,
        'num_indices_per_lookup_fixed': True,
        'numpy_rand_seed': 123,
        'rand_data_dist': "uniform",
        'rand_data_min': 1,
        'rand_data_max': 1,
        'rand_data_mu': -1,
        'rand_data_sigma': 1,
    })

    opt.ln_bot = np.fromstring(opt.arch_mlp_bot, dtype=int, sep="-")

    # Input and target at random
    opt.ln_emb = np.fromstring(opt.arch_embedding_size, dtype=int, sep="-")
    opt.m_den = opt.ln_bot[0]
    dataset = RandomDataset(opt)
    opt.nbatches = len(dataset.train_ld)

    opt.m_spa = opt.arch_sparse_feature_size
    num_fea = opt.ln_emb.size + 1  # num sparse + num dense features
    m_den_out = opt.ln_bot[opt.ln_bot.size - 1]
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out

    arch_mlp_top_adjusted = str(num_int) + "-" + opt.arch_mlp_top
    opt.ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    dlrm = DLRM_Net(
        opt.m_spa,
        opt.ln_emb,
        opt.ln_bot,
        opt.ln_top,
        arch_interaction_op=opt.arch_interaction_op,
        arch_interaction_itself=opt.arch_interaction_itself,
        sigmoid_bot=opt.sigmoid_bot,
        sigmoid_top=opt.sigmoid_top,
        sync_dense_params=opt.sync_dense_params,
        loss_threshold=opt.loss_threshold,
        ndevices=opt.ndevices,
        qr_flag=opt.qr_flag,
        qr_operation=opt.qr_operation,
        qr_collisions=opt.qr_collisions,
        qr_threshold=opt.qr_threshold,
        md_flag=opt.md_flag,
        md_threshold=opt.md_threshold,
    )

    X, lS_o, lS_i, targets = next(iter(dataset.train_ld))
    example_inputs = (X, lS_o, lS_i)

    runner = PyTorchRunner(dlrm, example_inputs=example_inputs, skip_script=True)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch_cuda(batch_size, num_runs, timeout):
    from utils.pytorch import PyTorchRunner

    def run_single_pass(torch_runner, dataset):
        output = torch_runner.run(batch_size, *example_inputs)
        dataset.submit_predictions(output)

    append_dlrm_to_pypath()
    from utils.recommendation.dlrm.dlrm_s_pytorch import DLRM_Net

    arch_embedding_size = "1000000-1000000-1000000-1000000-1000000-1000000-1000000-1000000"
    arch_sparse_feature_size = 64
    arch_mlp_bot = "512-512-64"
    arch_mlp_top = "1024-1024-1024-1"
    data_generation = "random"
    mini_batch_size = batch_size
    num_batches = 1
    num_indicies_per_lookup = 100

    opt = argparse.Namespace(**{
        'm_spa': None,
        'ln_emb': None,
        'ln_bot': None,
        'ln_top': None,
        'arch_interaction_op': "dot",
        'arch_interaction_itself': False,
        'sigmoid_bot': -1,
        'sigmoid_top': -1,
        'sync_dense_params': True,
        'ndevices': -1,
        'qr_flag': False,
        'qr_operation': "mult",
        'qr_collisions': 0,
        'qr_threshold': 200,
        'md_flag': False,
        'md_threshold': 200,
        'md_temperature': 0.3,
        'activation_function': "relu",
        'loss_function': "bce",
        'loss_weights': "1.0-1.0",
        'loss_threshold': 0.0,
        'round_targets': False,
        'data_size': 6,
        'data_generation': data_generation,
        'data_trace_file': "./input/dist_emb_j.log",
        'raw_data_file': "",
        'processed_data_file': "",
        'data_randomize': "total",
        'data_trace_enable_padding': False,
        'max_ind_range': -1,
        'num_workers': 0,
        'memory_map': False,
        'data_sub_sample_rate': 0.0,
        'learning_rate': 0.01,
        'lr_num_warmup_steps': 0,
        'lr_decay_start_step': 0,
        'lr_num_decay_steps': 0,
        'arch_embedding_size': arch_embedding_size,
        'arch_sparse_feature_size': arch_sparse_feature_size,
        'arch_mlp_bot': arch_mlp_bot,
        'arch_mlp_top': arch_mlp_top,
        'mini_batch_size': mini_batch_size,
        'num_batches': num_batches,
        'num_indices_per_lookup': num_indicies_per_lookup,
        'num_indices_per_lookup_fixed': True,
        'numpy_rand_seed': 123,
        'rand_data_dist': "uniform",
        'rand_data_min': 1,
        'rand_data_max': 1,
        'rand_data_mu': -1,
        'rand_data_sigma': 1,
    })

    opt.ln_bot = np.fromstring(opt.arch_mlp_bot, dtype=int, sep="-")

    # Input and target at random
    opt.ln_emb = np.fromstring(opt.arch_embedding_size, dtype=int, sep="-")
    opt.m_den = opt.ln_bot[0]
    dataset = RandomDataset(opt)
    opt.nbatches = len(dataset.train_ld)

    opt.m_spa = opt.arch_sparse_feature_size
    num_fea = opt.ln_emb.size + 1  # num sparse + num dense features
    m_den_out = opt.ln_bot[opt.ln_bot.size - 1]
    num_int = (num_fea * (num_fea - 1)) // 2 + m_den_out

    arch_mlp_top_adjusted = str(num_int) + "-" + opt.arch_mlp_top
    opt.ln_top = np.fromstring(arch_mlp_top_adjusted, dtype=int, sep="-")

    dlrm = DLRM_Net(
        opt.m_spa,
        opt.ln_emb,
        opt.ln_bot,
        opt.ln_top,
        arch_interaction_op=opt.arch_interaction_op,
        arch_interaction_itself=opt.arch_interaction_itself,
        sigmoid_bot=opt.sigmoid_bot,
        sigmoid_top=opt.sigmoid_top,
        sync_dense_params=opt.sync_dense_params,
        loss_threshold=opt.loss_threshold,
        ndevices=opt.ndevices,
        qr_flag=opt.qr_flag,
        qr_operation=opt.qr_operation,
        qr_collisions=opt.qr_collisions,
        qr_threshold=opt.qr_threshold,
        md_flag=opt.md_flag,
        md_threshold=opt.md_threshold,
    )

    X, lS_o, lS_i, targets = next(iter(dataset.train_ld))
    example_inputs = (X.cuda(), lS_o.cuda(), [i.cuda() for i in lS_i])

    runner = PyTorchRunner(dlrm.cuda(), disable_jit_freeze=True)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch_fp32(batch_size, num_runs, timeout):
    return run_pytorch_fp(batch_size, num_runs, timeout)


def main():
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
