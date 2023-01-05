# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import argparse

from utils.benchmark import run_model
from utils.nlp.wmt import WMT
from utils.misc import print_goodbye_message_and_die


def parse_args():
    parser = argparse.ArgumentParser(description="Run CTranslate model on WMT Translation task.")
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32", "fp16", "int8"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-f", "--framework",
                        type=str, choices=["ctranslate"], required=True,
                        help="specify the framework in which a model should be run")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--tokenizer_path",
                        type=str,
                        help="path to tokenizer (SentencePiece model)")
    parser.add_argument("--dataset_path",
                        type=str,
                        help="path to dataset file")
    parser.add_argument("--targets_path",
                        type=str,
                        help="path to file with target translations")
    parser.add_argument("--constant_input", action='store_true',
                        help="if true model will receive the same sentence as input every time")
    return parser.parse_args()


def run_ctranslate(model_path, batch_size, num_runs, timeout, tokenizer_path, dataset_path, targets_path, constant_input, **kwargs):
    from utils.ctranslate import CTranslateRunner

    def run_single_pass(ct_runner, dataset):
        output = ct_runner.run(tokenize(ct_runner.tokenizer, dataset.get_input_array()))

        for i in range(batch_size):
            dataset.submit_prediction(
                i,
                detokenize(ct_runner.tokenizer, output[i][0]['tokens'])
            )

    runner = CTranslateRunner(model_path, tokenizer_path)
    dataset = WMT(batch_size, dataset_path, targets_path, constant_input)

    def tokenize(tokenizer, sentence):
        return tokenizer.encode(sentence, out_type=str)
    
    def detokenize(tokenizer, tokens):
        return tokenizer.decode(tokens)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_ctranslate_fp32(model_path, batch_size, num_runs, timeout, tokenizer_path, dataset_path, targets_path, constant_input, **kwargs):
    return run_ctranslate(model_path, batch_size, num_runs, timeout, tokenizer_path, dataset_path, targets_path, constant_input, **kwargs)

def run_ctranslate_fp16(model_path, batch_size, num_runs, timeout, tokenizer_path, dataset_path, targets_path, constant_input, **kwargs):
    return run_ctranslate(model_path, batch_size, num_runs, timeout, tokenizer_path, dataset_path, targets_path, constant_input, **kwargs)

def run_ctranslate_int8(model_path, batch_size, num_runs, timeout, tokenizer_path, dataset_path, targets_path, constant_input, **kwargs):
    return run_ctranslate(model_path, batch_size, num_runs, timeout, tokenizer_path, dataset_path, targets_path, constant_input, **kwargs)


def main():
    args = parse_args()

    if args.framework == "ctranslate":
        if args.model_path is None:
            print_goodbye_message_and_die(
                "a path to model is unspecified!")
        elif args.precision == "fp32":
            run_ctranslate_fp32(**vars(args))
        elif args.precision == "fp16":
            run_ctranslate_fp16(**vars(args))
        elif args.precision == "int8":
            run_ctranslate_int8(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)
    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
