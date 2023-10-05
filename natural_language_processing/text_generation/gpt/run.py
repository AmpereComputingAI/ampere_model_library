from transformers import GPT2Tokenizer, GPT2Model

from utils.benchmark import run_model
from utils.nlp.text_generation_dummy import TextGenerationDummyDataset


def run_pytorch_fp32(model_name, batch_size, num_runs, timeout, **kwargs):
    from utils.pytorch import PyTorchRunner, PyTorchRunnerV2

    def run_single_pass(pytorch_runner, _dummy_dataset):
        output = pytorch_runner.run(1, _dummy_dataset.get_input()['input_ids'])
        _dummy_dataset.submit_count(batch_size)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def tokenize(text):
        return tokenizer(text, return_tensors='pt')

    model = GPT2Model.from_pretrained(model_name, torchscript=True)
    dataset = TextGenerationDummyDataset(batch_size, tokenize)

    runner = PyTorchRunnerV2(model)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser

    gpt_variants = ["gpt2", "gpt2-medium"]
    parser = DefaultArgParser(["pytorch"])
    parser.require_model_name(gpt_variants)
    parser.ask_for_batch_size()
    run_pytorch_fp32(**vars(parser.parse()))
