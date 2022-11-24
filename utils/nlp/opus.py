import random


class Opus:
    """
    A class providing facilities for preprocessing and postprocessing of Opus dataset.
    """

    def __init__(self, batch_size: int, tokenize_func, detokenize_func):
        self._batch_size = batch_size
        self._tokenize_func = tokenize_func
        self._detokenize_func = detokenize_func

        random.seed(12)
        self._iter = 0
        text = """translate English to French: Because participants were asked to pre-create some LLM prompts
        for their desired sub-tasks prior to the study, this may have unintentionally led to participants feeling 
        invested in their prompts and their particular chain decomposition, making them less inclined to consider other 
        chain structures or scrap the prompts they had already created. Yet, prior work in prototyping indicates that
        concurrently considering multiple alternatives (e.g., parallel prototyping [6]) can lead to better outcomes. """
        self._warm_up_txt = [text[:50] for _ in range(self._batch_size)]  # truncate input to 50 to make it short
        random_idxs = [random.choice(range(len(text) - 20)) for _ in range(self._batch_size)]
        self._txt = [f"translate English to French: {text[random_idxs[i]:random_idxs[i] + 20]}"
                     for i in range(self._batch_size)]

        self.available_instances = 1000

    def reset(self):
        return True

    def get_input(self):
        if self._iter < 2:
            self._iter += 1
            return self._tokenize_func(self._warm_up_txt)
        else:
            return self._tokenize_func(self._txt)

    def submit_prediction(self, output_tokens):
        self._detokenize_func(output_tokens)

    def summarize_accuracy(self):
        print(f"\nAccuracy check not yet implemented.")
        return {}
