import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

from utils.benchmark import run_model
from utils.nlp.lambada import Lambada


def run_pytorch_fp32(model_name, batch_size, num_runs, timeout, lambada_path, **kwargs):
    from utils.pytorch import PyTorchRunner, PyTorchRunnerV2, apply_jit_trace, apply_jit_script

    def run_single_pass(pytorch_runner, lambada):
        start_ids = lambada.get_input_array()[0]
        # output = pytorch_runner.run(None, start_ids, num_beams=2, no_repeat_ngram_size=2,
        #                             early_stopping=True, max_new_tokens=5)
        output = pytorch_runner.run(None, start_ids)
        pytorch_runner.set_task_size(output[1] - start_ids.shape[1])
        logits = output[0]
        token_ids = torch.argmax(logits, dim=-1)

        print(type(token_ids))
        print(token_ids)
        text = tokenizer.decode(token_ids)
        print(text)
        quit()

        output = detokenize(output[0])

        for i in range(batch_size):
            first_new_word = output.replace(detokenize(start_ids[0]), '').split()[0]
            lambada.submit_prediction(i, first_new_word)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def detokenize(answer):
        return tokenizer.decode(answer, skip_special_tokens=True)

    def tokenize(text):
        return tokenizer.encode(text, return_tensors='pt')

    model = GPT2LMHeadModel.from_pretrained(model_name, torchscript=True)
    model.eval()
    dataset = Lambada(batch_size, tokenize, detokenize, lambada_path)
    # model = apply_jit_trace(model, torch.randint(10000, (5,)))
    model = apply_jit_trace(model, (dataset.get_input_array()[0],))
    runner = PyTorchRunnerV2(model)
    # runner = PyTorchRunner(model, disable_jit_freeze=True, func="generate")

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser

    gpt_variants = ["gpt2"]
    parser = DefaultArgParser(["pytorch"])
    parser.require_model_name(gpt_variants)
    parser.ask_for_batch_size()
    parser.add_argument('--lambada_path', type=str, required=True, help="Path to Lambada dataset")
    run_pytorch_fp32(**vars(parser.parse()))
