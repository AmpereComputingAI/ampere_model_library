import argparse
from utils.tf import TFFrozenModelRunner
from utils.tflite import TFLiteRunner
import numpy as np
from utils.benchmark import run_model
from transformers import AutoTokenizer, TFAutoModelForQuestionAnswering
from utils.nlp.squad import Squad_v1_1


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
    parser.add_argument("--squad_path",
                        type=str,
                        help="path to directory with ImageNet validation images")
    return parser.parse_args()


def run_tf_fp32(model_path, batch_size, num_of_runs, timeout, squad_path):
    def run_single_pass(tf_runner, squad):
        # tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")
        # text = r"""
        # 🤗 Transformers (formerly known as pytorch-transformers and pytorch-pretrained-bert) provides general-purpose
        # architectures (BERT, GPT-2, RoBERTa, XLM, DistilBert, XLNet…) for Natural Language Understanding (NLU) and Natural
        # Language Generation (NLG) with over 32+ pretrained models in 100+ languages and deep interoperability between
        # TensorFlow 2.0 and PyTorch.
        # """
        # question = "How many pretrained models are available in 🤗 Transformers?"
        # inputs = tokenizer(question, text, add_special_tokens=True, return_tensors="np")
        # print(inputs)
        # ds
        # input_ids_raw = inputs['input_ids']
        # input_ids = np.pad(input_ids_raw, ((0, 0), (0, 384-input_ids_raw.shape[1])), 'constant', constant_values=0)
        # print(input_ids)
        # print(input_ids.shape)
        # input_mask = inputs['attention_mask']
        # input_mask = np.pad(input_mask, ((0, 0), (0, 384 - input_mask.shape[1])), 'constant', constant_values=0)
        # segment_ids = inputs['token_type_ids']
        # segment_ids = np.pad(segment_ids, ((0, 0), (0, 384 - segment_ids.shape[1])), 'constant', constant_values=0)
        # print(input_mask.shape)

        tf_runner.set_input_tensor("input_ids:0", squad.get_input_ids_array())
        tf_runner.set_input_tensor("input_mask:0", squad.get_attention_mask_array())
        tf_runner.set_input_tensor("segment_ids:0", squad.get_token_type_ids_array())

        output = tf_runner.run()

        for i in range(batch_size):
            answer_start_id, answer_end_id = np.argmax(output["logits:0"][i], axis=0)
            squad.submit_prediction(
                i,
                squad.extract_answer(i, answer_start_id, answer_end_id)
            )

    seq_size = 384
    tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased-whole-word-masking-finetuned-squad")

    def tokenize(question, text):
        return tokenizer(question, text, add_special_tokens=True)

    def detokenize(answer):
        return tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(answer))

    dataset = Squad_v1_1(batch_size, seq_size, tokenize, detokenize, seq_size, squad_path)
    runner = TFFrozenModelRunner(model_path, ["logits:0"])

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def main():
    args = parse_args()
    if args.precision == "fp32":
        run_tf_fp32(
            args.model_path, args.batch_size, args.num_runs, args.timeout, args.squad_path
        )
    else:
        assert False, f"Behaviour undefined for precision {args.precision}"


if __name__ == "__main__":
    main()
