from transformers import GPT2Tokenizer, GPT2LMHeadModel

from utils.benchmark import run_model
from utils.nlp.lambada import Lambada


def run_pytorch_fp32(model_name, batch_size, num_runs, timeout, lambada_path, **kwargs):
    from utils.pytorch import PyTorchRunnerV2, apply_jit_trace_module

    def run_single_pass(pytorch_runner, lambada):
        start_ids = lambada.get_input_array()[0]
        outputs = pytorch_runner.run(None, start_ids, do_sample=True, max_length=50, top_p=0.95)
        pytorch_runner.set_task_size(outputs.shape[1] - start_ids.shape[1])
        output = detokenize(outputs[0])

        for i in range(batch_size):
            first_new_word = output.replace(detokenize(start_ids[0]), '').split()[0]
            lambada.submit_prediction(i, first_new_word)

    tokenizer = GPT2Tokenizer.from_pretrained(model_name)

    def detokenize(answer):
        return tokenizer.decode(answer, skip_special_tokens=True)

    def tokenize(text):
        return tokenizer.encode(text, return_tensors='pt')

    model = GPT2LMHeadModel.from_pretrained(model_name, torchscript=True).eval()
    dataset = Lambada(batch_size, tokenize, detokenize, lambada_path)
    model.generate = apply_jit_trace_module(model, {"generate": dataset.get_input_array()[0]})
    runner = PyTorchRunnerV2(model.generate)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser

    gpt_variants = ["gpt2"]
    parser = DefaultArgParser(["pytorch"])
    parser.require_model_name(gpt_variants)
    parser.ask_for_batch_size()
    parser.add_argument('--lambada_path', type=str, required=True, help="Path to Lambada dataset")
    run_pytorch_fp32(**vars(parser.parse()))
