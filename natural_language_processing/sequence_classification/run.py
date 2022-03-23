import argparse
from utils.mrpc import MRPC
from utils.tf import NLPModelRunner
from utils.benchmark import run_model


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Hugging Face models "
                                                 "for Sequence Classification task on MRPC dataset")
    parser.add_argument("-m", "--model_name",
                        type=str, default="bert-base-cased-finetuned-mrpc",
                        help="name of the transformers model to run. "
                             "list of all available models is available at "
                             "https://huggingface.co/models")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--dataset_path",
                        type=str, required=True,
                        help="path to mrpc dataset. Original dataset can be downloaded from"
                             "https://www.microsoft.com/en-us/download/details.aspx?id=52398")
    return parser.parse_args()


def run(model_name, batch_size, num_runs, timeout, dataset_path):

    def run_single_pass(nlp_runner, mrpc):

        input, labels = mrpc.get_input_array()
        output = nlp_runner.run(input)

        predictions = mrpc.extract_prediction(output)
        for i in range(batch_size):
            mrpc.submit_predictions(
                int(predictions[i]),
                labels[i]
            )

    dataset = MRPC(model_name, batch_size, dataset_path)
    runner = NLPModelRunner(model_name)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def main():
    args = parse_args()
    run(args.model_name, args.batch_size, args.num_runs, args.timeout, args.dataset_path)


if __name__ == "__main__":
    main()

