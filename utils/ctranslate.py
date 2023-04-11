import time

import ctranslate2
import sentencepiece

import utils.benchmark as bench_utils
from utils.misc import advertise_aio
from utils.profiling import aio_profiler_enabled

class CTranslateRunner:
    """
    A class providing facilities to run CTranslate2 model.
    """

    def __init__(self, model, tokenizer, compute_type):
        # try:
        #     #TODO: Check for AIO
        # except AttributeError:
        #     utils.advertise_aio("CTranslate2")

        self.translator = ctranslate2.Translator(
            model, device='cpu', compute_type=compute_type,
            inter_threads=1, intra_threads=bench_utils.get_intra_op_parallelism_threads()
        )
        self.tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)

        self.__times_invoked = 0
        self.__start_times = list()
        self.__finish_times = list()
        self.is_profiling = aio_profiler_enabled()

        print("\nRunning with CTranslate2\n")

    def run(self, input):
        if self.__times_invoked == 2 and self.is_profiling:
            self.translator.init_profiling("cpu")
        start = time.time()
        outputs = self.translator.translate_batch(input)
        finish = time.time()

        self.__start_times.append(start)
        self.__finish_times.append(finish)
        self.__times_invoked += 1

        return outputs

    def print_performance_metrics(self, batch_size):
        """
        A function printing performance metrics on runs executed by the runner so far.
        :param batch_size: int, batch size - if batch size was varying over the runs an average should be supplied
        """
        
        if self.is_profiling:
            self.translator.dump_profiling()

        return bench_utils.print_performance_metrics(
            self.__start_times, self.__finish_times, self.__times_invoked, batch_size)
