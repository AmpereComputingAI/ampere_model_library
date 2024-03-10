# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import os
import unittest
import subprocess
import pathlib
import psutil
import torch
from utils.downloads.utils import get_downloads_path
from multiprocessing import Process, Queue


class LLaMA2(unittest.TestCase):
    def setUp(self):
        url = "https://github.com/tloen/alpaca-lora/raw/main/alpaca_data.json"
        self.dataset_path = pathlib.Path(get_downloads_path(), "alpaca_data.json")
        if not self.dataset_path.exists():
            subprocess.run(f"wget -P {get_downloads_path()} {url}".split(),
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @unittest.skipIf(psutil.virtual_memory().available / 1024 ** 3 < 100, "too little memory")
    @unittest.skipUnless('_aio_profiler_print' in dir(torch._C), "Ampere optimized PyTorch required")
    def test_llama2_7b(self):
        from natural_language_processing.text_generation.llama2.run import run_pytorch_fp32
        f1_ref = 0.290
        acc, _ = run_pytorch_fp32(model_name="meta-llama/Llama-2-7b-chat-hf",
                                  batch_size=1, num_runs=50, timeout=None, dataset_path=self.dataset_path)
        self.assertTrue(acc["f1"] / f1_ref > 0.95)

    @unittest.skipIf(psutil.virtual_memory().available / 1024 ** 3 < 150, "too little memory")
    @unittest.skipUnless('_aio_profiler_print' in dir(torch._C), "Ampere optimized PyTorch required")
    def test_llama2_13b(self):
        from natural_language_processing.text_generation.llama2.run import run_pytorch_fp32
        f1_ref = 0.164
        acc, _ = run_pytorch_fp32(model_name="meta-llama/Llama-2-13b-chat-hf",
                                  batch_size=1, num_runs=50, timeout=None, dataset_path=self.dataset_path)
        self.assertTrue(acc["f1"] / f1_ref > 0.95)


class Alpaca(unittest.TestCase):
    def setUp(self):
        url = "https://github.com/tloen/alpaca-lora/raw/main/alpaca_data.json"
        self.dataset_path = pathlib.Path(get_downloads_path(), "alpaca_data.json")
        if not self.dataset_path.exists():
            subprocess.run(f"wget -P {get_downloads_path()} {url}".split(),
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.model_path = pathlib.Path(get_downloads_path(), "alpaca_recovered")
        if not self.model_path.exists():
            url = os.environ.get("S3_URL_ALPACA_PYTORCH_FP32")
            assert url is not None
            subprocess.run(f"wget -P /tmp {url}".split(),
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(f"tar -xf /tmp/alpaca_recovered.tar.gz -C {get_downloads_path()}".split(),
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run("rm /tmp/alpaca_recovered.tar.gz".split(),
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @unittest.skipIf(psutil.virtual_memory().available / 1024 ** 3 < 100, "too little memory")
    @unittest.skipUnless('_aio_profiler_print' in dir(torch._C), "Ampere optimized PyTorch required")
    def test_alpaca(self):
        from natural_language_processing.text_generation.alpaca.run import run_pytorch_fp32
        exact_match_ref, f1_ref = 0.100, 0.317
        acc, _ = run_pytorch_fp32(model_path=self.model_path,
                                  batch_size=1, num_runs=50, timeout=None, dataset_path=self.dataset_path)
        self.assertTrue(acc["exact_match"] / exact_match_ref > 0.95)
        self.assertTrue(acc["f1"] / f1_ref > 0.95)


def run_process(wrapper, kwargs):
    output_queue = Queue()
    kwargs.update({"q": output_queue})
    p = Process(target=wrapper, kwargs=kwargs)
    p.start()
    p.join()
    return output_queue.get()


class Whisper(unittest.TestCase):
    def setUp(self):
        from speech_recognition.whisper.run import run_pytorch_fp32

        def wrapper(**kwargs):
            kwargs["q"].put(run_pytorch_fp32(**kwargs)[0])

        self.wrapper = wrapper

    @unittest.skipIf(psutil.virtual_memory().available / 1024 ** 3 < 50, "too little memory")
    def test_whisper_tiny_en(self):
        wer_ref = 0.155
        acc = run_process(self.wrapper, {"model_name": "tiny.en", "num_runs": 30, "timeout": None})
        self.assertTrue(wer_ref / acc["wer_score"] > 0.95)

    @unittest.skipIf(psutil.virtual_memory().available / 1024 ** 3 < 100, "too little memory")
    def test_whisper_large(self):
        wer_ref = 0.124
        acc = run_process(self.wrapper, {"model_name": "large", "num_runs": 30, "timeout": None})
        self.assertTrue(wer_ref / acc["wer_score"] > 0.95)


class DLRM(unittest.TestCase):
    def setUp(self):
        self.dataset_path = pathlib.Path(get_downloads_path(), "criteo_preprocessed")
        if not self.dataset_path.exists():
            url = os.environ.get("S3_URL_CRITEO_DATASET")
            assert url is not None
            subprocess.run(f"wget -P /tmp {url}".split(),
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(f"tar -xf /tmp/criteo_preprocessed.tar.gz -C {get_downloads_path()}".split(),
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run("rm /tmp/criteo_preprocessed.tar.gz".split(),
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.model_path = pathlib.Path(get_downloads_path(), "tb0875_10M.pt")
        if not self.model_path.exists():
            subprocess.run(
                f"wget -P {get_downloads_path()} "
                f"{'https://dlrm.s3-us-west-1.amazonaws.com/models/tb0875_10M.pt'}".split(),
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    @unittest.skipIf(psutil.virtual_memory().available / 1024 ** 3 < 100, "too little memory")
    def test_dlrm_debug(self):
        from recommendation.dlrm.run import run_pytorch_fp32

        def wrapper(**kwargs):
            kwargs["q"].put(run_pytorch_fp32(**kwargs)[0])

        auc_ref = 0.583
        acc = run_process(wrapper, {"model_path": self.model_path, "dataset_path": self.dataset_path,
                                    "batch_size": 2048, "num_runs": 30, "timeout": None, "debug": True})
        self.assertTrue(acc["auc"] / auc_ref > 0.95)


if __name__ == "__main__":
    unittest.main()
