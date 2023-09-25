import argparse

from transformers import GPT2Tokenizer, GPT2Model

from utils.benchmark import run_model
from utils.misc import print_goodbye_message_and_die
from utils.nlp.text_generation_dummy import TextGenerationDummyDataset

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
    from utils.pytorch import PyTorchRunner

    def run_single_pass(pytorch_runner, _dummy_dataset):
        output = pytorch_runner.run(1, _dummy_dataset.get_input()['input_ids'])
        _dummy_dataset.submit_count(batch_size)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def tokenize(text):
        return tokenizer(text, return_tensors='pt')

    model = GPT2Model.from_pretrained(model_name, torchscript=True)
    dataset = TextGenerationDummyDataset(batch_size, tokenize)

    runner = PyTorchRunner(model,
                           disable_jit_freeze=disable_jit_freeze,
                           skip_script=True,
                           example_inputs=(dataset.get_input()['input_ids'],))

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


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
