# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
import unittest
import subprocess
import pathlib
import psutil
from utils.downloads.utils import get_downloads_path


class LLaMA2(unittest.TestCase):
    def setUp(self):
        url = "https://github.com/tloen/alpaca-lora/raw/main/alpaca_data.json"
        self.dataset_path = pathlib.Path(get_downloads_path(), "alpaca_data.json")
        if not self.dataset_path.exists():
            subprocess.run(f"wget -P {get_downloads_path()} {url}".split())

    @unittest.skipIf(psutil.virtual_memory().available / 1024 ** 3 < 100,
                     "too little memory")
    def test_llama2_7b(self):
        from natural_language_processing.text_generation.llama2.run import run_pytorch_fp32
        f1_ref = 0.290
        acc, perf = run_pytorch_fp32(model_name="meta-llama/Llama-2-7b-chat-hf",
                                     batch_size=1, num_runs=50, timeout=None, dataset_path=self.dataset_path)
        self.assertTrue(acc["f1"] / f1_ref > 0.95)

    @unittest.skipIf(psutil.virtual_memory().available / 1024 ** 3 < 100,
                     "too little memory")
    def test_llama2_13b(self):
        from natural_language_processing.text_generation.llama2.run import run_pytorch_fp32
        f1_ref = 0.164
        acc, perf = run_pytorch_fp32(model_name="meta-llama/Llama-2-13b-chat-hf",
                                     batch_size=1, num_runs=50, timeout=None, dataset_path=self.dataset_path)
        self.assertTrue(acc["f1"] / f1_ref > 0.95)


if __name__ == "__main__":
    unittest.main()
