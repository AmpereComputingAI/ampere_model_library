from natural_language_processing.text_generation.transformers.src.transformers.models.auto.modeling_auto import \
    AutoModelForCausalLM
from natural_language_processing.text_generation.transformers.src.transformers.models.auto.tokenization_auto import \
    AutoTokenizer
from utils.nlp.alpaca_instruct import AlpacaInstruct
from utils.pytorch import PyTorchRunnerV2, apply_compile
from utils.benchmark import run_model


def run_pytorch(model_path, num_runs, timeout, dataset_path):
    def run_single_pass(pytorch_runner, _dataset):
        inputs = encode(_dataset.get_input_string())
        outputs = pytorch_runner.run(inputs=inputs.input_ids, max_new_tokens=100)
        pytorch_runner.set_task_size(outputs.shape[1] - inputs.input_ids.shape[1])
        response = decode(outputs[:, inputs.input_ids.shape[1]:])
        _dataset.submit_prediction(response)

    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.merge_qkv()
    model.eval()
    model.greedy_search = apply_compile(model.greedy_search)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    dataset = AlpacaInstruct(1, dataset_path=dataset_path)
    encode = lambda i: tokenizer(i, return_tensors="pt")
    decode = lambda t: tokenizer.batch_decode(t, skip_special_tokens=True)[0]

    runner = PyTorchRunnerV2(model.generate)

    return run_model(run_single_pass, runner, dataset, 1, num_runs, timeout)


def run_pytorch_fp32(model_path, num_runs, timeout, dataset_path, **kwargs):
    return run_pytorch(model_path, num_runs, timeout, dataset_path)


def main():
    from utils.helpers import DefaultArgParser
    parser = DefaultArgParser(["pytorch"])
    parser.require_model_path()
    parser.add_argument("--dataset_path",
                        type=str,
                        help="path to JSON file with instructions")
    run_pytorch_fp32(**vars(parser.parse()))


if __name__ == "__main__":
    main()
