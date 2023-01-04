import csv
import json
import os
import time

import ctranslate2
import sentencepiece

import utils.benchmark as bench_utils
from utils.misc import advertise_aio

class CTranslateRunner:
    """
    A class providing facilities to run CTranslate2 model.
    """

    def __init__(self, model, tokenizer):
        # try:
        #     #TODO: Check for AIO
        # except AttributeError:
        #     utils.advertise_aio("CTranslate2")

        self.translator = ctranslate2.Translator(model, device='cpu', compute_type='float', inter_threads=1, intra_threads=bench_utils.get_intra_op_parallelism_threads())
        self.tokenizer = sentencepiece.SentencePieceProcessor(tokenizer)

        self.__times_invoked = 0
        self.__start_times = list()
        self.__finish_times = list()

        print("\nRunning with CTranslate2\n")

    def run(self, input):
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
        perf = bench_utils.print_performance_metrics(
            self.__start_times, self.__finish_times, self.__times_invoked, batch_size)
        if os.getenv("AIO_PROFILER", "0") == "1":
            ctranslate2.AIO.print_profile_data() #TODO: See if it works

        dump_dir = os.environ.get("RESULTS_DIR")
        if dump_dir is not None and len(self.__start_times) > 2:
            with open(f"{dump_dir}/meta_{os.getpid()}.json", "w") as f:
                json.dump({"batch_size": batch_size}, f)
            with open(f"{dump_dir}/{os.getpid()}.csv", "w") as f:
                writer = csv.writer(f)
                writer.writerow(self.__start_times[2:])
                writer.writerow(self.__finish_times[2:])

        return perf
