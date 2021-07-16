from transformers import TensorFlowBenchmark, TensorFlowBenchmarkArguments
import argparse
import tensorflow as tf
import utils.benchmark as bench_utils


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark NLP models")
    parser.add_argument("-m", "--model", nargs='+', type=str, required=False, default=["bert-base-cased"],
                        help="you can see the available list of models here: "
                        "https://huggingface.co/transformers/pretrained_models.html")
    parser.add_argument("-b", "--batch_size", nargs='+', default=[8],
                        type=int, required=False,
                        help="batch size to feed the model with")
    parser.add_argument("--sequence_length", nargs='+', required=False,
                        type=int, default=[8],
                        help="sequence length to feed the model with")
    return parser.parse_args()


def benchmark_nlp_model(model, batch_size, sequence_length):

    tf.config.threading.set_intra_op_parallelism_threads(bench_utils.get_intra_op_parallelism_threads())
    tf.config.threading.set_inter_op_parallelism_threads(1)

    args = TensorFlowBenchmarkArguments(models=model, batch_sizes=batch_size,
                                        sequence_lengths=sequence_length, memory=False)

    benchmark = TensorFlowBenchmark(args)
    results = benchmark.run()

    # for k, v in results.time_inference_result.items():
    #
    #     result = v.get("result")
    #     batch_size = v.get("bs")
    #     sequence_length = v.get("ss")
    #
    #     print('\n')
    #     print(f'for model {k}:')
    #     for i in batch_size:
    #         for j in sequence_length:
    #             inference_time = result.get(i).get(j)
    #
    #             print(f"for {i} batch size and {j} sequence length, the latency is {i/inference_time}")


def main():
    args = parse_args()
    benchmark_nlp_model(args.model, args.batch_size, args.sequence_length)


if __name__ == "__main__":
    main()
