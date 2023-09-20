import torch
from transformers import GPT2Tokenizer, GPT2Model

from utils.misc import print_goodbye_message_and_die


def parse_args():
    parser = argparse.ArgumentParser(description="Run model from Hugging Face's transformers repository"
                                                 "for text generation task.")
    parser.add_argument("-m", "--model_name",
                        type=str, choices=["gpt2"], required=True,
                        help="name of the model")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("-f", "--framework",
                        type=str, choices=["pytorch"], required=True,
                        help="specify the framework in which a model should be run")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    return parser.parse_args()


def run_pytorch(model_name, batch_size, num_runs, timeout, disable_jit_freeze=False, **kwargs):

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2Model.from_pretrained(model_name, torchscript=True)

    text = "Hi, how are you?"
    encoded_input = tokenizer(text, return_tensors='pt')
    input_dict = {key: value for key, value in encoded_input.items()}

    traced_model = torch.jit.trace(model, (input_dict['input_ids'],))
    #traced_model = torch.jit.trace(model, (encoded_input,))
    frozen_model = torch.jit.freeze(traced_model)

    #output = frozen_model(**encoded_input)
    output = frozen_model(input_dict['input_ids'])

    def run_single_pass(pytorch_runner, squad):
        pass

    model = AutoModelForQuestionAnswering.from_pretrained(model_name, torchscript=True)
    dataset = Squad_v1_1(batch_size, tokenize, detokenize, dataset_path=squad_path)

    runner = PyTorchRunner(model,
                           disable_jit_freeze=disable_jit_freeze,
                           example_inputs=[val for val in dataset.get_input_arrays().values()])


def run_pytorch_fp32(model_name, batch_size, num_runs, timeout, disable_jit_freeze, **kwargs):
    return run_pytorch(model_name, batch_size, num_runs, timeout, disable_jit_freeze, **kwargs)


def main():
    args = parse_args()

    if args.framework == "pytorch":
        run_pytorch(**vars(args))
    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
