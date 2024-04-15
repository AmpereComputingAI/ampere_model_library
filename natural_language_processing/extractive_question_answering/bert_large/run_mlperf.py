# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC

import argparse
import numpy as np
from utils.benchmark import run_model
from utils.nlp.squad import Squad_v1_1
from utils.misc import print_goodbye_message_and_die, download_squad_1_1_dataset


def parse_args():
    parser = argparse.ArgumentParser(description="Run BERT Large model (from mlcommons:inference repo).")
    parser.add_argument("-m", "--model_path",
                        type=str,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32", "fp16"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str, default="tf",
                        choices=["tf", "pytorch"],
                        help="specify the framework in which a model should be run")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--squad_path",
                        type=str,
                        help="path to directory with ImageNet validation images")
    parser.add_argument("--disable_jit_freeze", action='store_true',
                        help="if true model will be run not in jit freeze mode")
    return parser.parse_args()


def run_tf_fp(model_path, batch_size, num_runs, timeout, squad_path):
    from transformers import AutoTokenizer
    from utils.tf import TFFrozenModelRunner

    def run_single_pass(tf_runner, squad):

        tf_runner.set_input_tensor("input_ids:0", squad.get_input_ids_array())
        tf_runner.set_input_tensor("input_mask:0", squad.get_attention_mask_array())
        tf_runner.set_input_tensor("segment_ids:0", squad.get_token_type_ids_array())

        output = tf_runner.run(batch_size * seq_size)

        for i in range(batch_size):
            answer_start_id, answer_end_id = np.argmax(output["logits:0"][i], axis=0)
            squad.submit_prediction(
                i,
                squad.extract_answer(i, answer_start_id, answer_end_id)
            )

    seq_size = 384
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    def tokenize(question, text):
        return tokenizer(question, text, add_special_tokens=True, truncation=True, max_length=seq_size)

    def detokenize(answer):
        return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer))

    dataset = Squad_v1_1(batch_size, tokenize, detokenize, seq_size, squad_path)
    runner = TFFrozenModelRunner(model_path, ["logits:0"])

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_tf_fp32(model_path, batch_size, num_runs, timeout, squad_path, **kwargs):
    return run_tf_fp(model_path, batch_size, num_runs, timeout, squad_path)


def run_tf_fp16(model_path, batch_size, num_runs, timeout, squad_path, **kwargs):
    return run_tf_fp(model_path, batch_size, num_runs, timeout, squad_path)


def run_pytorch_fp(model_path, batch_size, num_runs, timeout, squad_path, disable_jit_freeze=False):
    from transformers import AutoTokenizer, BertConfig, BertForQuestionAnswering
    import torch
    from utils.pytorch import PyTorchRunner

    def run_single_pass(pytorch_runner, squad):
        input_tensor = squad.get_input_arrays()
        output = pytorch_runner.run(batch_size * input_tensor["input_ids"].size()[1], **dict(input_tensor))

        for i in range(batch_size):
            answer_start_id = output[0][i].argmax()
            answer_end_id = output[1][i].argmax()
            squad.submit_prediction(
                i,
                squad.extract_answer(i, answer_start_id, answer_end_id)
            )

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad",
        padding=True, truncation=True, model_max_length=512)

    def tokenize(question, text):
        return tokenizer(question, text, padding=True, truncation=True, return_tensors="pt")

    def detokenize(answer):
        return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer))

    bert_config = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "type_vocab_size": 2,
        "vocab_size": 30522,
        "torchscript": True
    }
    config = BertConfig(**bert_config)
    model = BertForQuestionAnswering(config)
    model.load_state_dict(torch.load(model_path), strict=False)
    dataset = Squad_v1_1(batch_size, tokenize, detokenize, dataset_path=squad_path)
    runner = PyTorchRunner(model,
                           disable_jit_freeze=disable_jit_freeze,
                           example_inputs=[val for val in dataset.get_input_arrays().values()])

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch_cuda(model_path, batch_size, num_runs, timeout, squad_path, disable_jit_freeze=False, **kwargs):
    import torch
    from utils.pytorch import PyTorchRunner
    from transformers import AutoTokenizer, BertConfig, BertForQuestionAnswering

    def run_single_pass(pytorch_runner, squad):
        input_tensor = squad.get_input_arrays()
        output = pytorch_runner.run(batch_size * input_tensor["input_ids"].size()[1], **{k: v.cuda() for k, v in input_tensor.items()})

        for i in range(batch_size):
            answer_start_id = output[0][i].argmax()
            answer_end_id = output[1][i].argmax()
            squad.submit_prediction(
                i,
                squad.extract_answer(i, answer_start_id, answer_end_id)
            )

    tokenizer = AutoTokenizer.from_pretrained(
        "bert-large-uncased-whole-word-masking-finetuned-squad", padding=True, truncation=True, model_max_length=512)

    def tokenize(question, text):
        return tokenizer(question, text, padding=True, truncation=True, return_tensors="pt")

    def detokenize(answer):
        return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer))

    bert_config = {
        "attention_probs_dropout_prob": 0.1,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.1,
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "max_position_embeddings": 512,
        "num_attention_heads": 16,
        "num_hidden_layers": 24,
        "type_vocab_size": 2,
        "vocab_size": 30522,
        "torchscript": False
    }
    config = BertConfig(**bert_config)
    model = BertForQuestionAnswering(config)
    model.load_state_dict(torch.load(model_path), strict=False)
    dataset = Squad_v1_1(batch_size, tokenize, detokenize, dataset_path=squad_path)
    runner = PyTorchRunner(model.cuda(), disable_jit_freeze=True)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch_fp32(model_path, batch_size, num_runs, timeout, squad_path, disable_jit_freeze, **kwargs):
    return run_pytorch_fp(model_path, batch_size, num_runs, timeout, squad_path, disable_jit_freeze)


def main():
    args = parse_args()
    download_squad_1_1_dataset()

    if args.framework == "tf":
        if args.batch_size > 1:
            print_goodbye_message_and_die("This model supports only BS=1")
            
        if args.model_path is None:
            print_goodbye_message_and_die("a path to model is unspecified!")

        if args.precision == "fp32":
            run_tf_fp32(**vars(args))
        elif args.precision == "fp16":
            run_tf_fp16(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)
    elif args.framework == "pytorch":
        if args.model_path is None:
            print_goodbye_message_and_die(
                "a path to model is unspecified!")

        import torch
        if torch.cuda.is_available():
            run_pytorch_cuda(**vars(args))
        elif args.precision == "fp32":
            run_pytorch_fp32(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)
    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
