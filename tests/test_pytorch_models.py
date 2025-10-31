# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2025, Ampere Computing LLC
import os
import signal
import time
import unittest
import subprocess
import pathlib
import psutil
import torch
from utils.downloads.utils import get_downloads_path
from multiprocessing import Process, Queue

TIMEOUT = 3 * 60 * 60
pid = os.getpid()


def run_process(wrapper, kwargs):
    def wrapper_outer(**kwargs):
        try:
            wrapper(**kwargs)
        except Exception as e:
            print(f"\nException encountered: {e}")
            os.kill(pid, signal.SIGTERM)

    start = time.time()
    output_queue = Queue()
    kwargs.update({"q": output_queue})
    p = Process(target=wrapper_outer, kwargs=kwargs)
    p.start()
    output = output_queue.get(block=True, timeout=max(0, int(TIMEOUT - (time.time() - start))))
    p.join(timeout=max(0, int(TIMEOUT - (time.time() - start))))
    return output


class LLaMA2(unittest.TestCase):
    def setUp(self):
        from natural_language_processing.text_generation.llama2.run import run_pytorch_fp32

        url = "https://github.com/tloen/alpaca-lora/raw/main/alpaca_data.json"
        self.dataset_path = pathlib.Path(get_downloads_path(), "alpaca_data.json")
        if not self.dataset_path.exists():
            subprocess.run(f"wget -P {get_downloads_path()} {url}".split(),
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def wrapper(**kwargs):
            kwargs["q"].put(run_pytorch_fp32(**kwargs)[0])

        self.wrapper = wrapper

    # @unittest.skipIf(psutil.virtual_memory().available / 1024 ** 3 < 100, "too little memory")
    # @unittest.skipUnless('_aio_profiler_print' in dir(torch._C), "Ampere optimized PyTorch required")
    # def test_llama2_7b(self):
    #     f1_ref = 0.330
    #     acc = run_process(self.wrapper,
    #                       {"model_name": "meta-llama/Llama-2-7b-chat-hf", "batch_size": 1, "num_runs": 50,
    #                        "timeout": None, "dataset_path": self.dataset_path})
    #     self.assertTrue(acc["f1"] / f1_ref > 0.95)
    #
    # @unittest.skipIf(psutil.virtual_memory().available / 1024 ** 3 < 200, "too little memory")
    # @unittest.skipUnless('_aio_profiler_print' in dir(torch._C), "Ampere optimized PyTorch required")
    # def test_llama2_13b(self):
    #     f1_ref = 0.261
    #     acc = run_process(self.wrapper,
    #                       {"model_name": "meta-llama/Llama-2-13b-chat-hf", "batch_size": 1, "num_runs": 50,
    #                        "timeout": None, "dataset_path": self.dataset_path})
    #     self.assertTrue(acc["f1"] / f1_ref > 0.95)


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

    # @unittest.skipIf(psutil.virtual_memory().available / 1024 ** 3 < 100, "too little memory")
    # @unittest.skipUnless('_aio_profiler_print' in dir(torch._C), "Ampere optimized PyTorch required")
    # def test_alpaca(self):
    #     from natural_language_processing.text_generation.alpaca.run import run_pytorch_fp32
    #
    #     def wrapper(**kwargs):
    #         kwargs["q"].put(run_pytorch_fp32(**kwargs)[0])
    #
    #     exact_match_ref, f1_ref = 0.220, 0.547
    #     acc = run_process(wrapper, {"model_path": self.model_path, "batch_size": 1, "num_runs": 50,
    #                                 "timeout": None, "dataset_path": self.dataset_path})
    #     self.assertTrue(acc["exact_match"] / exact_match_ref > 0.95)
    #     self.assertTrue(acc["f1"] / f1_ref > 0.95)


class Whisper(unittest.TestCase):
    def setUp(self):
        def wrapper_openai(**kwargs):
            from speech_recognition.whisper.run import run_pytorch_fp32
            kwargs["q"].put(run_pytorch_fp32(**kwargs)[0])

        def wrapper_hf(**kwargs):
            from speech_recognition.whisper.run_hf import run_pytorch_fp32
            kwargs["q"].put(run_pytorch_fp32(**kwargs)[0])

        self.wrapper_openai = wrapper_openai
        self.wrapper_hf = wrapper_hf

    @unittest.skipIf(psutil.virtual_memory().available / 1024 ** 3 < 50, "too little memory")
    def test_whisper_tiny_en(self):
        wer_ref = 0.155
        acc = run_process(self.wrapper_openai, {"model_name": "tiny.en", "num_runs": 30, "timeout": None})
        self.assertTrue(wer_ref / acc["wer_score"] > 0.95)

    @unittest.skipIf(psutil.virtual_memory().available / 1024 ** 3 < 50, "too little memory")
    def test_whisper_hf_tiny_en(self):
        wer_ref = 0.111
        acc = run_process(self.wrapper_hf, {"model_name": "openai/whisper-tiny.en", "num_runs": 18,
                                            "batch_size": 4, "timeout": None})
        self.assertTrue(wer_ref / acc["wer_score"] > 0.95)

    @unittest.skipIf(psutil.virtual_memory().available / 1024 ** 3 < 100, "too little memory")
    @unittest.skipUnless('_aio_profiler_print' in dir(torch._C), "too slow to run with native")
    def test_whisper_large(self):
        wer_ref = 0.124
        acc = run_process(self.wrapper_openai, {"model_name": "large", "num_runs": 30, "timeout": None})
        self.assertTrue(wer_ref / acc["wer_score"] > 0.95)


class WhisperTranslate(unittest.TestCase):
    def setUp(self):
        from speech_recognition.whisper_translate.run import run_pytorch_fp32

        self.dataset_path = pathlib.Path(get_downloads_path(), "covost2_ja")
        if not self.dataset_path.exists():
            url = os.environ.get("S3_URL_COVOST2_DATASET")
            assert url is not None
            subprocess.run(f"mkdir {self.dataset_path}".split(),
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(f"wget -P /tmp {url}".split(),
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(f"tar -xf /tmp/covost2_ja.tar -C {self.dataset_path}".split(),
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run("rm /tmp/covost2_ja.tar".split(),
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        def wrapper(**kwargs):
            kwargs["q"].put(run_pytorch_fp32(**kwargs)[0])

        self.wrapper = wrapper

    # @unittest.skipIf(psutil.virtual_memory().available / 1024 ** 3 < 100, "too little memory")
    # @unittest.skipUnless('_aio_profiler_print' in dir(torch._C), "too slow to run with native")
    # def test_whisper_translate_medium(self):
    #     wer_ref = 0.475
    #     acc = run_process(self.wrapper, {"model_name": "large", "num_runs": 30, "timeout": None,
    #                                      "dataset_path": self.dataset_path})
    #     self.assertTrue(wer_ref / acc["bleu_score"] > 0.95)


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

    # @unittest.skipIf(psutil.virtual_memory().available / 1024 ** 3 < 100, "too little memory")
    # def test_dlrm_debug(self):
    #     from recommendation.dlrm.run import run_pytorch_fp32
    #
    #     def wrapper(**kwargs):
    #         kwargs["q"].put(run_pytorch_fp32(**kwargs)[0])
    #
    #     auc_ref = 0.583
    #     acc = run_process(wrapper, {"model_path": self.model_path, "dataset_path": self.dataset_path,
    #                                 "batch_size": 2048, "num_runs": 30, "timeout": None, "debug": True})
    #     self.assertTrue(acc["auc"] / auc_ref > 0.95)


class BERT(unittest.TestCase):
    def setUp(self):
        self.dataset_path = pathlib.Path(get_downloads_path(), "dev-v1.1.json")
        if not self.dataset_path.exists():
            subprocess.run("wget -P /tmp https://data.deepai.org/squad1.1.zip".split(),
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run(f"unzip /tmp/squad1.1.zip -d {get_downloads_path()}".split(),
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            subprocess.run("rm /tmp/squad1.1.zip".split(),
                           check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.model_path = pathlib.Path(get_downloads_path(), "bert_large_mlperf.pt")
        if not self.model_path.exists():
            subprocess.run(
                f"wget -O {str(self.model_path)} "
                f"{'https://zenodo.org/records/3733896/files/model.pytorch?download=1'}".split(),
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # def test_bert_large_mlperf(self):
    #     from natural_language_processing.extractive_question_answering.bert_large.run_mlperf import run_pytorch_fp32
    #
    #     def wrapper(**kwargs):
    #         kwargs["q"].put(run_pytorch_fp32(**kwargs)[0])
    #
    #     exact_match_ref, f1_ref = 0.750, 0.817
    #     acc = run_process(wrapper, {"model_path": self.model_path, "squad_path": self.dataset_path,
    #                                 "batch_size": 1, "num_runs": 24, "timeout": None,
    #                                 "fixed_input_size": None, "disable_jit_freeze": False})
    #     self.assertTrue(acc["exact_match"] / exact_match_ref > 0.95)
    #     self.assertTrue(acc["f1"] / f1_ref > 0.95)


def download_imagenet_maybe():
    dataset_path = pathlib.Path(get_downloads_path(), "ILSVRC2012_onspecta")
    if not dataset_path.exists():
        url = os.environ.get("S3_URL_IMAGENET_DATASET")
        assert url is not None
        subprocess.run(f"wget -P /tmp {url}".split(),
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(f"tar -xf /tmp/ILSVRC2012_onspecta.tar.gz -C {get_downloads_path()}".split(),
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run("rm /tmp/ILSVRC2012_onspecta.tar.gz".split(),
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    labels_path = pathlib.Path(get_downloads_path(), "imagenet_labels_onspecta.txt")
    if not labels_path.exists():
        url = os.environ.get("S3_URL_IMAGENET_DATASET_LABELS")
        assert url is not None
        subprocess.run(f"wget -P {get_downloads_path()} {url}".split(),
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dataset_path, labels_path


class DenseNet(unittest.TestCase):
    def setUp(self):
        self.dataset_path, self.labels_path = download_imagenet_maybe()

    # def test_densenet_121(self):
    #     from computer_vision.classification.densenet_121.run import run_pytorch_fp32
    #
    #     def wrapper(**kwargs):
    #         kwargs["q"].put(run_pytorch_fp32(**kwargs)[0])
    #
    #     top_1_ref, top_5_ref = 0.717, 0.905
    #     acc = run_process(wrapper, {"model_name": "densenet121", "images_path": self.dataset_path,
    #                                 "labels_path": self.labels_path, "batch_size": 32, "num_runs": 10, "timeout": None,
    #                                 "disable_jit_freeze": False})
    #     self.assertTrue(acc["top_1_acc"] / top_1_ref > 0.95)
    #     self.assertTrue(acc["top_5_acc"] / top_5_ref > 0.95)


class Inception(unittest.TestCase):
    def setUp(self):
        self.dataset_path, self.labels_path = download_imagenet_maybe()

    # def test_inception_v3(self):
    #     from computer_vision.classification.inception_v3.run import run_pytorch_fp32
    #
    #     def wrapper(**kwargs):
    #         kwargs["q"].put(run_pytorch_fp32(**kwargs)[0])
    #
    #     top_1_ref, top_5_ref = 0.765, 0.932
    #     acc = run_process(wrapper, {"model_name": "inception_v3", "images_path": self.dataset_path,
    #                                 "labels_path": self.labels_path, "batch_size": 32, "num_runs": 10, "timeout": None,
    #                                 "disable_jit_freeze": False})
    #     self.assertTrue(acc["top_1_acc"] / top_1_ref > 0.95)
    #     self.assertTrue(acc["top_5_acc"] / top_5_ref > 0.95)


class ResNet(unittest.TestCase):
    def setUp(self):
        self.dataset_path, self.labels_path = download_imagenet_maybe()

    def test_resnet_50_v15(self):
        from computer_vision.classification.resnet_50_v15.run import run_pytorch_fp32

        def wrapper(**kwargs):
            kwargs["q"].put(run_pytorch_fp32(**kwargs)[0])

        top_1_ref, top_5_ref = 0.717, 0.904
        acc = run_process(wrapper, {"model_name": "resnet50", "images_path": self.dataset_path,
                                    "labels_path": self.labels_path, "batch_size": 32, "num_runs": 10, "timeout": None})
        self.assertTrue(acc["top_1_acc"] / top_1_ref > 0.95)
        self.assertTrue(acc["top_5_acc"] / top_5_ref > 0.95)

        print('here-resnet')
    print('here-resnet1')


class VGG(unittest.TestCase):
    def setUp(self):
        self.dataset_path, self.labels_path = download_imagenet_maybe()

    # def test_vgg16(self):
    #     from computer_vision.classification.vgg_16.run import run_pytorch_fp32
    #
    #     def wrapper(**kwargs):
    #         kwargs["q"].put(run_pytorch_fp32(**kwargs)[0])
    #
    #     top_1_ref, top_5_ref = 0.661, 0.896
    #     acc = run_process(wrapper, {"model_name": "vgg16", "images_path": self.dataset_path,
    #                                 "labels_path": self.labels_path, "batch_size": 32,
    #                                 "num_runs": 10, "timeout": None})
    #     self.assertTrue(acc["top_1_acc"] / top_1_ref > 0.95)
    #     self.assertTrue(acc["top_5_acc"] / top_5_ref > 0.95)


def download_coco_maybe():
    dataset_path = pathlib.Path(get_downloads_path(), "COCO2014_onspecta")
    if not dataset_path.exists():
        url = os.environ.get("S3_URL_COCO_DATASET")
        assert url is not None
        subprocess.run(f"wget -P /tmp {url}".split(),
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(f"tar -xf /tmp/COCO2014_onspecta.tar.gz -C {get_downloads_path()}".split(),
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run("rm /tmp/COCO2014_onspecta.tar.gz".split(),
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    annotations_path = pathlib.Path(get_downloads_path(), "COCO2014_anno_onspecta.json")
    if not annotations_path.exists():
        url = os.environ.get("S3_URL_COCO_DATASET_ANNOTATIONS")
        assert url is not None
        subprocess.run(f"wget -P {get_downloads_path()} {url}".split(),
                       check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return dataset_path, annotations_path


class YOLO(unittest.TestCase):
    def setUp(self):
        self.dataset_path, self.annotations_path = download_coco_maybe()
        self.yolo_v5_m_path = pathlib.Path(get_downloads_path(), "yolov5mu.pt")
        if not self.yolo_v5_m_path.exists():
            subprocess.run(
                f"wget -P {get_downloads_path()} "
                f"{'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov5mu.pt'}".split(),
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.yolo_v8_s_path = pathlib.Path(get_downloads_path(), "yolov8s.pt")
        if not self.yolo_v8_s_path.exists():
            subprocess.run(
                f"wget -P {get_downloads_path()} "
                f"{'https://github.com/ultralytics/assets/releases/download/v8.1.0/yolov8s.pt'}".split(),
                check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # def test_yolo_v5_m(self):
    #     from computer_vision.object_detection.yolo_v5.run import run_pytorch_fp32
    #
    #     def wrapper(**kwargs):
    #         kwargs["q"].put(run_pytorch_fp32(**kwargs)[0])
    #
    #     coco_map_ref = 0.492
    #     acc = run_process(wrapper, {"model_path": self.yolo_v5_m_path, "images_path": self.dataset_path,
    #                                 "anno_path": self.annotations_path, "batch_size": 1, "num_runs": 100,
    #                                 "timeout": None, "disable_jit_freeze": False})
    #     self.assertTrue(acc["coco_map"] / coco_map_ref > 0.95)

    # def test_yolo_v8_s(self):
    #     from computer_vision.object_detection.yolo_v8.run import run_pytorch_fp32
    #     from utils.benchmark import set_global_intra_op_parallelism_threads
    #     set_global_intra_op_parallelism_threads(32)
    #
    #     def wrapper(**kwargs):
    #         kwargs["q"].put(run_pytorch_fp32(**kwargs)[0])
    #
    #     coco_map_ref = 0.353
    #     acc = run_process(wrapper, {"model_path": self.yolo_v8_s_path, "images_path": self.dataset_path,
    #                                 "anno_path": self.annotations_path, "batch_size": 1, "num_runs": 465,
    #                                 "timeout": None, "disable_jit_freeze": False})
    #     self.assertTrue(acc["coco_map"] / coco_map_ref > 0.95)


if __name__ == "__main__":
    unittest.main()
