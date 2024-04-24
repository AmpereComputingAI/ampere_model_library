# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import argparse
import torch
from utils.benchmark import run_model
import utils.speech_recognition.rnnt.model as rnnt
from utils.misc import print_goodbye_message_and_die
from utils.speech_recognition.libri_speech import LibriSpeech
from utils.speech_recognition.rnnt.decoders import ScriptGreedyDecoder
from utils.speech_recognition.rnnt.config import config as rnnt_config


def parse_args():
    parser = argparse.ArgumentParser(description="Run RNNT model.")
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
                        type=str,
                        choices=["pytorch"], required=True,
                        help="specify the framework in which a model should be run")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--dataset_path",
                        type=str,
                        help="path to directory with LibriSpeech dataset")
    parser.add_argument("--disable_jit_freeze", action='store_true',
                        help="if true model will be run not in jit freeze mode")
    return parser.parse_args()


def run_pytorch_fp32(model_path, batch_size, num_runs, timeout, dataset_path, disable_jit_freeze):
    from utils.pytorch import PyTorchRunner

    def run_single_pass(pytorch_runner, libri_speech):

        input_arrays = libri_speech.get_input_arrays()
        _, _, output = pytorch_runner.run(batch_size, input_arrays)
        libri_speech.submit_predictions(output)

    rnnt_vocab = rnnt_config['labels']['labels']
    featurizer_config = rnnt_config['input_eval']

    model = rnnt.RNNT(feature_config=featurizer_config,
                      rnnt=rnnt_config['rnnt'],
                      num_classes=len(rnnt_vocab))

    model.load_state_dict(load_and_migrate_checkpoint(model_path), strict=True)
    decoder = ScriptGreedyDecoder(len(rnnt_vocab) - 1, model)
    runner = PyTorchRunner(decoder, disable_jit_freeze=disable_jit_freeze)
    dataset = LibriSpeech(dataset_dir_path=dataset_path, config=rnnt_config, max_batch_size=batch_size)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def load_and_migrate_checkpoint(ckpt_path):
    checkpoint = torch.load(ckpt_path, map_location="cpu")
    migrated_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        key = key.replace("joint_net", "joint.net")
        migrated_state_dict[key] = value
    del migrated_state_dict["audio_preprocessor.featurizer.fb"]
    del migrated_state_dict["audio_preprocessor.featurizer.window"]
    return migrated_state_dict


def main():
    args = parse_args()
    if args.framework == "pytorch":
        if args.model_path is None:
            print_goodbye_message_and_die(
                "a path to model is unspecified!")

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
