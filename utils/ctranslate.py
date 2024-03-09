# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import time
import ctranslate2
import sentencepiece
from utils.benchmark import Runner, get_intra_op_parallelism_threads
from utils.profiling import aio_profiler_enabled


class CTranslateRunner(Runner):
    """
    A class providing facilities to run CTranslate2 model.
    """

    def __init__(self, model, tokenizer, compute_type, throughput_only=False):
        # try:
        #     #TODO: Check for AIO
        # except AttributeError:
        #     utils.advertise_aio("CTranslate2")

        super().__init__(throughput_only)
        self.translator = ctranslate2.Translator(
            model, device='cpu', compute_type=compute_type,
            inter_threads=1, intra_threads=get_intra_op_parallelism_threads()
        )
        self.tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)
        self.is_profiling = aio_profiler_enabled()

        print("\nRunning with CTranslate2\n")

    def run(self, task_size=None, *args, **kwargs):
        if self._times_invoked == 2 and self.is_profiling:
            self.translator.init_profiling("cpu")
        start = time.time()
        outputs = self.translator.translate_batch(input)
        finish = time.time()

        self._start_times.append(start)
        self._finish_times.append(finish)
        self.set_task_size(task_size)
        self._times_invoked += 1

        return outputs

    def print_performance_metrics(self):
        """
        A function printing performance metrics on runs executed by the runner so far.
        """

        if self.is_profiling:
            self.translator.dump_profiling()

        return self.print_metrics()
