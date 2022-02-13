import argparse

import numpy as np

from utils.tf import TFFrozenModelRunner
from utils.benchmark import run_model
from transformers import AutoTokenizer
from utils.nlp.squad import Squad_v1_1
from utils.misc import print_goodbye_message_and_die


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
                        choices=["tf"],
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
    return parser.parse_args()


def run_tf_fp(model_path, batch_size, num_runs, timeout, squad_path):
    def run_single_pass(tf_runner, squad):

        tf_runner.set_input_tensor("input_ids:0", squad.get_input_ids_array())
        tf_runner.set_input_tensor("input_mask:0", squad.get_attention_mask_array())
        tf_runner.set_input_tensor("segment_ids:0", squad.get_token_type_ids_array())

        output = tf_runner.run()

        for i in range(batch_size):
            answer_start_id, answer_end_id = np.argmax(output["logits:0"][i], axis=0)
            squad.submit_prediction(
                i,
                squad.extract_answer(i, answer_start_id, answer_end_id)
            )

    seq_size = 384
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    def tokenize(question, text):
        return tokenizer(question, text[:seq_size], add_special_tokens=True)

    def detokenize(answer):
        return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer))

    dataset = Squad_v1_1(batch_size, tokenize, detokenize, seq_size, squad_path)
    runner = TFFrozenModelRunner(model_path, ["logits:0"])

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_tf_fp32(model_path, batch_size, num_runs, timeout, squad_path, **kwargs):
    return run_tf_fp(model_path, batch_size, num_runs, timeout, squad_path)


def run_tf_fp16(model_path, batch_size, num_runs, timeout, squad_path, **kwargs):
    return run_tf_fp(model_path, batch_size, num_runs, timeout, squad_path)


def main():
    args = parse_args()
    if args.framework == "tf":
        if args.model_path is None:
            print_goodbye_message_and_die(
                "a path to model is unspecified!")

        if args.precision == "fp32":
            run_tf_fp32(**vars(args))
        elif args.precision == "fp16":
            run_tf_fp16(**vars(args))
        else:
            print_goodbye_message_and_die(
                "this model seems to be unsupported in a specified precision: " + args.precision)

    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
