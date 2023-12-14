import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.benchmark import run_model
from utils.nlp.lambada import Lambada


def run_pytorch_fp32(model_name, batch_size, num_runs, timeout, lambada_path, **kwargs):
    from utils.pytorch import PyTorchRunnerV2, apply_compile_maybe

    def run_single_pass(pytorch_runner, lambada):
        start_ids = lambada.get_input_array()[0]
        output = pytorch_runner.run(inputs=start_ids, max_new_tokens=10)
        pytorch_runner.set_task_size(output.shape[1] - start_ids.shape[1])
        output = detokenize(output[0])

        for i in range(batch_size):
            first_new_word = output.replace(detokenize(start_ids[0]), '').split()[0]
            lambada.submit_prediction(i, first_new_word)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def detokenize(answer):
        return tokenizer.decode(answer, skip_special_tokens=True)

    def tokenize(text):
        return tokenizer.encode(text, return_tensors='pt')

    model = AutoModelForCausalLM.from_pretrained(model_name).eval()
    dataset = Lambada(batch_size, tokenize, detokenize, lambada_path)
    aio = '_aio_profiler_print' in dir(torch._C) and os.environ.get("AIO_PROCESS_MODE") != "0"
    model.greedy_search = apply_compile_maybe(model.greedy_search, aio)

    runner = PyTorchRunnerV2(model.generate)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser

    gpt_variants = ["EleutherAI/gpt-j-6B"]
    parser = DefaultArgParser(["pytorch"])
    parser.require_model_name(gpt_variants)
    parser.ask_for_batch_size()
    parser.add_argument('--lambada_path', type=str, required=True, help="Path to Lambada dataset")
    run_pytorch_fp32(**vars(parser.parse()))
