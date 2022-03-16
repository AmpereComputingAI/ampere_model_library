import argparse

import numpy as np
import torch

from utils.pytorch import PyTorchRunner
from utils.benchmark import run_model
from transformers import AutoTokenizer, AutoModelForTokenClassification
from utils.nlp.conll2003 import CoNNL2003
from utils.misc import print_goodbye_message_and_die


def parse_args():
    parser = argparse.ArgumentParser(description="Run model from Huggingface's transformers repo for summarization task.")
    parser.add_argument("-m", "--model_name",
                        type=str, default="dslim/bert-large-NER",
                        help="name of the model")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
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
    parser.add_argument("--conll2003_path",
                        type=str,
                        help="path to directory with the CNN/DailyMail dataset")
    return parser.parse_args()


def run_pytorch(model_name, batch_size, num_runs, timeout, conll2003_path, **kwargs):

    def run_single_pass(pytorch_runner, conll2003):
        
        input = torch.tensor(np.array(conll2003.get_input_ids_array(), dtype=np.int32))
        output = pytorch_runner.run((input))
        
        for i in range(batch_size):
            tokens = tokenizer.convert_ids_to_tokens(input[i])
            token_predictions = [id2label[int(x)] for x in np.argmax(output.logits[i], axis=1)]

            wp_preds = list(zip(tokens, token_predictions))
            prediction = []
            for token_pred, mapping in zip(wp_preds, conll2003.get_offset_mapping_array()[i]):
                if mapping[0] == 0 and mapping[1] != 0:
                    prediction.append(token_pred[1])
                else:
                    continue
            
            conll2003.submit_prediction(
                i,
                prediction)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(text):
        return tokenizer(text, return_offsets_mapping=True, is_split_into_words=True)

    def detokenize(summary):
        return tokenizer.decode(summary)

    dataset = CoNNL2003(batch_size, tokenize, detokenize, dataset_path=conll2003_path)
    model = AutoModelForTokenClassification.from_pretrained(model_name)
    id2label = model.config.id2label
    runner = PyTorchRunner(model, disable_jit_freeze=True)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)

def main():
    args = parse_args()
    if args.framework == "pytorch":
        run_pytorch(**vars(args))
    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
