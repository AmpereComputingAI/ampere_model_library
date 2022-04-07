import argparse
import torch

import numpy as np
#import tensorflow as tf

from utils.tf import TFSavedModelRunner
from utils.pytorch import PyTorchRunner
from utils.benchmark import run_model
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, TFAutoModelForQuestionAnswering, BertModel, BertTokenizer, BertForQuestionAnswering, AutoConfig
from utils.nlp.squad import Squad_v1_1
from utils.misc import print_goodbye_message_and_die


def parse_args():
    parser = argparse.ArgumentParser(description="Run model from Huggingface's transformers repo for extractive question answering task.")
    parser.add_argument("-m", "--model_name",
                        type=str, default="bert-large-uncased-whole-word-masking-finetuned-squad",
                        help="name of the model")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str,
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
    return parser.parse_args()


def run_tf(model_name, batch_size, num_runs, timeout, squad_path, **kwargs):

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
    runner.model = tf.function(TFAutoModelForQuestionAnswering.from_pretrained(model_name))

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch(model_name, batch_size, num_runs, timeout, squad_path, disable_jit_freeze=False, **kwargs):

    def run_single_pass(pytorch_runner, squad):
        inputs = squad.get_input_arrays()
        print(inputs)
        #sf
        output = pytorch_runner.run(dict(inputs))
        #print(output.start_logits)
        #print(output[1].shape)
        #print(output)
        #print(output.start_logits[0].argmax())
        #print(np.argmax(output[0][0]))
        #print(np.argmax(output[1][0]))
        #fsddfs

        for i in range(batch_size):
            #answer_start_id = output.start_logits[i].argmax()
            #answer_end_id = output.end_logits[i].argmax()
            answer_start_id = output[0][i].argmax()
            answer_end_id = output[1][i].argmax()
            squad.submit_prediction(
                i,
                squad.extract_answer(i, answer_start_id, answer_end_id)
            )

    tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)
    #tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    def tokenize(question, text):
        print(question)
        print(text)
        return tokenizer(question, text, return_tensors="pt")#, add_special_tokens=True)

    def detokenize(answer):
        #x = tokenizer.decode(answer)
        x = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer))
        print(x)
        print("______________________")
        return x

    model = AutoModelForQuestionAnswering.from_pretrained(model_name, torchscript=True)
    dataset = Squad_v1_1(batch_size, tokenize, detokenize, dataset_path=squad_path)
    #print(dataset.get_input_arrays())
    example_inputs_tmp = dataset.get_input_arrays()
    example_inputs = [val for val in example_inputs_tmp.values()]
    
    runner = PyTorchRunner(model,
                           disable_jit_freeze=disable_jit_freeze,
                           example_inputs=example_inputs)
                           #example_inputs=dataset.get_input_arrays())
    #dataset.reset()

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def main():
    args = parse_args()
    if args.framework == "tf":
        run_tf(**vars(args))
    elif args.framework == "pytorch":
        run_pytorch(**vars(args))
    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
