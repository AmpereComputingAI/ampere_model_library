# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import argparse
from utils.nlp.mrpc import MRPC
from utils.benchmark import run_model
from utils.misc import print_goodbye_message_and_die


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Hugging Face models "
                                                 "for Sequence Classification task on MRPC dataset")
    parser.add_argument("-m", "--model_name",
                        type=str, choices=["bert-base-cased-finetuned-mrpc"], required=True,
                        help="name of the transformers model to run. "
                             "list of all available models is available at "
                             "https://huggingface.co/models")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str, choices=["tf"], required=True,
                        help="specify the framework in which a model should be run")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--dataset_path",
                        type=str,
                        help="path to mrpc dataset. Original dataset can be downloaded from"
                             "https://www.microsoft.com/en-us/download/details.aspx?id=52398")
    return parser.parse_args()


def run_tf(model_name, batch_size, num_runs, timeout, dataset_path):
    import tensorflow as tf
    from transformers import TFAutoModelForSequenceClassification
    from utils.tf import TFSavedModelRunner

    def run_single_pass(nlp_runner, mrpc):

        input, labels = mrpc.get_input_array()
        output = nlp_runner.run(batch_size, input)
        predictions = mrpc.extract_prediction(output)

        for i in range(batch_size):
            mrpc.submit_predictions(
                predictions[i],
                labels[i]
            )

    dataset = MRPC(model_name, batch_size, dataset_path)

    runner = TFSavedModelRunner()
    runner.model = tf.function(TFAutoModelForSequenceClassification.from_pretrained(model_name))

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def main():
    args = parse_args()
    if args.framework == "tf":
        run_tf(**vars(args))
    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()

