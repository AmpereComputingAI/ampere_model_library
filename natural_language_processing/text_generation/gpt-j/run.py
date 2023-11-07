from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.benchmark import run_model
from utils.nlp.lambada import Lambada


def run_pytorch_fp32(model_name, batch_size, num_runs, timeout, lambada_path, **kwargs):
    from utils.pytorch import PyTorchRunner, PyTorchRunnerV2

    def run_single_pass(pytorch_runner, lambada):
        start_ids = lambada.get_input_array()[0]
        output = pytorch_runner.run(None, start_ids, num_beams=2, no_repeat_ngram_size=2,
                                    early_stopping=True, max_new_tokens=5)
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

    model = AutoModelForCausalLM.from_pretrained(model_name, pad_token_id=tokenizer.eos_token_id)
    dataset = Lambada(batch_size, tokenize, detokenize, lambada_path)
    runner = PyTorchRunner(model, disable_jit_freeze=True,
                           example_inputs=dataset.get_input_array()[0], func="generate")
    # runner = PyTorchRunner(model, disable_jit_freeze=False, func="generate")
    # runner = PyTorchRunnerV2(model)

    return run_model(run_single_pass, runner, dataset, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser

    gpt_variants = ["EleutherAI/gpt-j-6B"]
    parser = DefaultArgParser(["pytorch"])
    parser.require_model_name(gpt_variants)
    parser.ask_for_batch_size()
    parser.add_argument('--lambada_path', type=str, required=True, help="Path to Lambada dataset")
    run_pytorch_fp32(**vars(parser.parse()))
