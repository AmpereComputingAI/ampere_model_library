import argparse
import numpy as np
import tensorflow as tf
from utils.tf import TFSavedModelRunner
from utils.benchmark import run_model
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
from utils.nlp.squad import Squad_v1_1


def parse_args():
    parser = argparse.ArgumentParser(description="Run model from Huggingface's transformers repo for extractive question answering task.")
    parser.add_argument("-m", "--model_name",
                        type=str, default="bert-large-uncased-whole-word-masking-finetuned-squad",
                        help="name of the model")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
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


def run_tf(model_name, batch_size, num_of_runs, timeout, squad_path):
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
        return tokenizer(question, text, add_special_tokens=True)

    def detokenize(answer):
        return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer))

    dataset = Squad_v1_1(batch_size, tokenize, detokenize, dataset_path=squad_path)
    runner = TFSavedModelRunner()
    runner.model = TFAutoModelForQuestionAnswering.from_pretrained(model_name)

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def main():
    args = parse_args()
    run_tf(
            args.model_name, args.batch_size, args.num_runs, args.timeout, args.squad_path
        )


if __name__ == "__main__":
    main()
