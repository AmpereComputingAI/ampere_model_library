import argparse

import transformers

from utils.nlp.alpaca_instruct import AlpacaInstruct
from utils.pytorch import PyTorchRunner
from utils.benchmark import run_model
from utils.misc import print_goodbye_message_and_die


def parse_args():
    parser = argparse.ArgumentParser(description="Run karpathy/nanoGPT with small modifications.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=True,
                        help="path to the model")
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
    parser.add_argument("--dataset_path",
                        type=str,
                        help="path to JSON file with instructions")
    parser.add_argument("--disable_jit_freeze", action='store_true',
                        help="if true model will be run not in jit freeze mode")
    return parser.parse_args()


def run_pytorch(model_path, batch_size, num_runs, timeout, dataset_path, disable_jit_freeze=False, **kwargs):

    def run_single_pass(pytorch_runner, dataset):
        inputs = encode(dataset.get_input_array())
        outputs = pytorch_runner.run({"inputs": inputs.input_ids, "max_new_tokens": 100})
        num_tokens = outputs.shape[1] - inputs.input_ids.shape[1]
        response = decode(outputs[:, inputs.input_ids.shape[1]:])
        dataset.submit_prediction(response)


    model = transformers.AutoModelForCausalLM.from_pretrained(model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    dataset = AlpacaInstruct(batch_size, dataset_path=dataset_path)
    encode = lambda i: tokenizer(i, return_tensors="pt")
    decode = lambda t: tokenizer.batch_decode(t, skip_special_tokens=True)[0]

    runner = PyTorchRunner(model, disable_jit_freeze=disable_jit_freeze, func="generate")

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch_fp32(model_path, batch_size, num_runs, timeout, dataset_path, disable_jit_freeze=False, **kwargs):
    run_pytorch(model_path, batch_size, num_runs, timeout, dataset_path, disable_jit_freeze, **kwargs)


def main():
    args = parse_args()
    if args.framework == "pytorch":
        if args.batch_size != 1:
            raise ValueError("Batch size must be 1 for this model.")
        run_pytorch_fp32(**vars(args))
    else:
        print_goodbye_message_and_die(
            "this model seems to be unsupported in a specified framework: " + args.framework)


if __name__ == "__main__":
    main()
