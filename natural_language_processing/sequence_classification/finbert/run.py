import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from utils.pytorch import PyTorchRunnerV2
from utils.benchmark import run_model
from utils.nlp.financial_phrasebank import FinancialPhraseBank


def run_pytorch(num_runs, timeout):
    def run_single_pass(pytorch_runner, dataset):
        input_tensor = tokenizer.encode(dataset.get_input_string(), truncation=True, return_tensors="pt")
        preds = softmax(pytorch_runner.run(input_tensor)[0])
        pytorch_runner.set_task_size(input_tensor.nelement())
        dataset.submit_prediction(preds)

    model_name = "ProsusAI/finbert"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, torchscript=True)
    model.eval()
    model = torch.jit.freeze(torch.jit.trace(model, tokenizer.encode("Stocks rallied and the British pound gained.",
                                                                     return_tensors='pt')))

    softmax = torch.nn.Softmax(dim=1)
    runner = PyTorchRunnerV2(model)

    return run_model(run_single_pass, runner, FinancialPhraseBank(), 1, num_runs, timeout)


def run_pytorch_fp32(num_runs, timeout, **kwargs):
    return run_pytorch(num_runs, timeout)


def main():
    from utils.helpers import DefaultArgParser
    parser = DefaultArgParser(["pytorch"])
    run_pytorch_fp32(**vars(parser.parse()))


if __name__ == "__main__":
    main()
