import argparse
from utils.mrpc import MRPC
from utils.tf import NLPModelRunner
from utils.benchmark import run_model
from utils.profiling import set_profile_path
import tensorflow as tf
from utils.profiling import get_profile_path

MODEL_NAME = "bert-base-cased-finetuned-mrpc"


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate the Hugging Face models for GLUE tasks dataset")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int, default=1,
                        help="number of passes through network to execute")
    parser.add_argument("--dataset_path",
                        type=str, required=True,
                        help="path to mrpc dataset. Original dataset can be downloaded from"
                             "https://www.microsoft.com/en-us/download/details.aspx?id=52398")
    return parser.parse_args()


def run(batch_size, num_runs, timeout, dataset_path):

    def run_single_pass(nlp_runner, mrpc):

        input, labels = mrpc.get_input_array()
        output = nlp_runner.run(input)

        predictions = mrpc.extract_prediction(output)
        for i in range(batch_size):
            mrpc.submit_predictions(
                predictions[i],
                labels[i]
            )

    dataset = MRPC(MODEL_NAME, batch_size, dataset_path)
    runner = NLPModelRunner(MODEL_NAME)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def main():
    args = parse_args()
    set_profile_path(MODEL_NAME)

    # options = tf.profiler.experimental.ProfilerOptions(
    #     host_tracer_level=3,
    #     python_tracer_level=0,
    #     device_tracer_level=0
    # )
    # test = '/onspecta/dev/mz/natural_language_processing'
    # tf.profiler.experimental.start(get_profile_path(), options=options)

    run(args.batch_size, args.num_runs, args.timeout, args.dataset_path)

    tf.profiler.experimental.stop()


if __name__ == "__main__":
    main()

