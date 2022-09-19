# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import argparse

import numpy as np
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TFAutoModelForQuestionAnswering

from utils.benchmark import run_model
from utils.nlp.squad import Squad_v1_1
from utils.misc import print_goodbye_message_and_die, download_squad_1_1_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run model from Huggingface's transformers repo for extractive"
                                                 "question answering task.")
    parser.add_argument("-m", "--model_name",
                        type=str, choices=["jimypbr/bert-base-uncased-squad",
                                           "madlag/bert-base-uncased-squadv1-x2.44-f87.7-d26-hybrid-filled-v1",
                                           "salti/bert-base-multilingual-cased-finetuned-squad"], required=True,
                        help="name of the model")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str, choices=["tf", "pytorch"], required=True,
                        help="specify the framework in which a model should be run")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--squad_path",
                        type=str,
                        help="path to .json file with Squad v1.1 data")
    return parser.parse_args()


def run_tf(model_name, batch_size, num_runs, timeout, squad_path, **kwargs):
    import tensorflow as tf
    from utils.tf import TFSavedModelRunner

    def run_single_pass(tf_runner, squad):
        output = tf_runner.run(np.array(squad.get_input_ids_array(), dtype=np.int32))

        for i in range(batch_size):
            answer_start_id = np.argmax(output.start_logits[i])
            answer_end_id = np.argmax(output.end_logits[i])
            squad.submit_prediction(
                i,
                squad.extract_answer(i, answer_start_id, answer_end_id)
            )

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(question, text):
        return tokenizer(question, text, padding=True, pad_to_multiple_of=32, truncation=True)

    def detokenize(answer):
        return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer))

    dataset = Squad_v1_1(batch_size, tokenize, detokenize, dataset_path=squad_path)
    runner = TFSavedModelRunner()
    runner.model = tf.function(TFAutoModelForQuestionAnswering.from_pretrained(model_name))

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch(model_name, batch_size, num_runs, timeout, squad_path, disable_jit_freeze=False, **kwargs):
    from utils.pytorch import PyTorchRunner

    def run_single_pass(pytorch_runner, squad):
        output = pytorch_runner.run(dict(squad.get_input_arrays()))

        for i in range(batch_size):
            answer_start_id = output[0][i].argmax()
            answer_end_id = output[1][i].argmax()
            squad.submit_prediction(
                i,
                squad.extract_answer(i, answer_start_id, answer_end_id)
            )

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding=True, truncation=True, model_max_length=512)

    def tokenize(question, text):
        return tokenizer(question, text, padding=True, truncation=True, return_tensors="pt")

    def detokenize(answer):
        return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer))

    model = AutoModelForQuestionAnswering.from_pretrained(model_name, torchscript=True)
    dataset = Squad_v1_1(batch_size, tokenize, detokenize, dataset_path=squad_path)

    runner = PyTorchRunner(model,
                           disable_jit_freeze=disable_jit_freeze,
                           example_inputs=[val for val in dataset.get_input_arrays().values()])

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_tf_fp32(model_name, batch_size, num_runs, timeout, squad_path, **kwargs):
    return run_tf(model_name, batch_size, num_runs, timeout, squad_path, **kwargs)


def run_pytorch_fp32(model_name, batch_size, num_runs, timeout, squad_path, disable_jit_freeze, **kwargs):
    return run_pytorch(model_name, batch_size, num_runs, timeout, squad_path, disable_jit_freeze, **kwargs)


def main():
    args = parse_args()
    download_squad_1_1_dataset()

    if args.framework == "tf":
        run_tf(**vars(args))
    elif args.framework == "pytorch":

        run_pytorch(**vars(args))
    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
