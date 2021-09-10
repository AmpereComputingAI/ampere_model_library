import argparse
from utils.tf import TFFrozenModelRunner
from utils.tflite import TFLiteRunner
import numpy as np
from utils.benchmark import run_model
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering


def parse_args():
    parser = argparse.ArgumentParser(description="Run MobileNet v2 model.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=True,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--images_path",
                        type=str,
                        help="path to directory with ImageNet validation images")
    parser.add_argument("--labels_path",
                        type=str,
                        help="path to file with validation labels")
    return parser.parse_args()


def run_tf_fp32(model_path, batch_size, num_of_runs, timeout, images_path, labels_path):

    def run_single_pass(tf_runner, imagenet):
        tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        text = r"""
        🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
        architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
        Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
        TensorFlow 2.0 and PyTorch.
        """
        question = "How many pretrained models are available in 🤗 Transformers?"
        inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="tf")
        seq_size = 384
        input_ids_raw = inputs['input_ids'].numpy()
        input_ids = np.pad(input_ids_raw, ((0, 0), (0, 384-input_ids_raw.shape[1])), 'constant', constant_values=0)
        print(input_ids)
        print(input_ids.shape)
        input_mask = inputs['attention_mask'].numpy()
        input_mask = np.pad(input_mask, ((0, 0), (0, 384 - input_mask.shape[1])), 'constant', constant_values=0)
        segment_ids = inputs['token_type_ids'].numpy()
        segment_ids = np.pad(segment_ids, ((0, 0), (0, 384 - segment_ids.shape[1])), 'constant', constant_values=0)
        print(input_mask.shape)

        tf_runner.set_input_tensor("input_ids:0", input_ids)#squad.get_input_ids_array(seq_size))
        tf_runner.set_input_tensor("input_mask:0", input_mask)#squad.get_input_mask_array(seq_size))
        tf_runner.set_input_tensor("segment_ids:0", segment_ids)#squad.get_segment_ids_array(seq_size))
        output = tf_runner.run()
        print(output['logits:0'][0])
        start_end = np.argmax(output['logits:0'][0], axis=0)
        answer_start = start_end[0]
        answer_end = start_end[1] + 1
        print(answer_start)
        answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(input_ids_raw[0][answer_start:answer_end]))
        print(f"Question: {question}")
        print(f"Answer: {answer}")
        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output["MobilenetV2/Predictions/Reshape_1:0"][i]),
                imagenet.extract_top5(output["MobilenetV2/Predictions/Reshape_1:0"][i])
            )

    dataset = None#Squad(batch_size, "RGB", images_path, labels_path,
                    #   pre_processing="Inception", is1001classes=True)
    runner = TFFrozenModelRunner(model_path, ["logits:0"])

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def main():
    args = parse_args()
    if args.precision == "fp32":
        run_tf_fp32(
            args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.labels_path
        )
    else:
        assert False, f"Behaviour undefined for precision {args.precision}"


if __name__ == "__main__":
    main()
