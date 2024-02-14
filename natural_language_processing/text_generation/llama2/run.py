from utils.pytorch import PyTorchRunnerV2, apply_compile
from utils.benchmark import run_model
from utils.nlp.alpaca_instruct import AlpacaInstruct
from transformers import LlamaForCausalLM, AutoTokenizer


def run_pytorch(model_name, batch_size, num_runs, timeout, dataset_path):
    def run_single_pass(pytorch_runner, _dataset):
        input_tensor = tokenizer.encode(_dataset.get_input_string(), return_tensors="pt")
        input_tensor = torch.cat([input_tensor for _ in range(batch_size)], 0)
        output = pytorch_runner.run(inputs=input_tensor, max_length=400)
        pytorch_runner.set_task_size(sum([len(output[i]) - len(input_tensor[i]) for i in range(batch_size)]))
        for i in range(batch_size):
            _dataset.submit_prediction(tokenizer.decode(output[i][len(input_tensor[i]):], skip_special_tokens=True))

    import numpy as np
    import torch
    np.random.seed(44)
    torch.manual_seed(44)

    model = LlamaForCausalLM.from_pretrained(model_name, torchscript=True)
    model.merge_qkv()
    model.eval()
    model.generate = apply_compile(model.generate)

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left')
    tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})

    dataset = AlpacaInstruct(batch_size, dataset_path=dataset_path)

    runner = PyTorchRunnerV2(model.generate, throughput_only=True)
    from transformers.generation.utils import aml_runner
    global aml_runner
    aml_runner = runner

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


def run_pytorch_fp32(model_name, batch_size, num_runs, timeout, dataset_path, **kwargs):
    return run_pytorch(model_name, batch_size, num_runs, timeout, dataset_path)


def main():
    from utils.helpers import DefaultArgParser
    llama_variants = ["meta-llama/Llama-2-7b-chat-hf", "meta-llama/Llama-2-13b-chat-hf"]
    parser = DefaultArgParser(["pytorch"])
    parser.ask_for_batch_size()
    parser.require_model_name(llama_variants)
    parser.add_argument("--dataset_path",
                        type=str,
                        help="path to JSON file with instructions")
    run_pytorch_fp32(**vars(parser.parse()))


if __name__ == "__main__":
    main()
