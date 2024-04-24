# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC


def run_pytorch_fp32(model_name, num_runs, timeout, dataset_path, **kwargs):
    import os
    import sys
    import torch
    batch_size = 1
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "whisper"))
    from utils.benchmark import run_model
    from utils.misc import print_warning_message
    from utils.pytorch import PyTorchRunnerV2
    from utils.speech_recognition.covost2 import Covost2
    from speech_recognition.whisper.whisper.whisper import load_model
    from speech_recognition.whisper.whisper.whisper.transcribe import transcribe
    model = load_model(model_name)
    model.eval()

    def single_pass_pytorch(_runner, _covost2):
        array = _covost2.get_input_array()
        audio = torch.from_numpy(array.astype("float32"))
        _covost2.submit_translation(
            _runner.run(batch_size * array.shape[0], audio)["text"].lstrip().replace(".", "")
        )

    def translate_wrapper(audio):
        return transcribe(model, audio, verbose=None, task="translate", language="ja")

    runner = PyTorchRunnerV2(translate_wrapper, throughput_only=True)
    librispeech = Covost2(dataset_path)
    print_warning_message("Sampling rate Whisper operates at is 16,000 Hz, therefore throughput values below can be "
                          "divided by 16,000 to derive 'seconds of processed audio per second'")
    return run_model(single_pass_pytorch, runner, librispeech, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser
    whisper_variants = ["tiny", "base", "small", "medium", "large"]
    parser = DefaultArgParser(["pytorch"])
    parser.require_model_name(whisper_variants)
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="path to the CommonVoice Japanese dataset directory")
    run_pytorch_fp32(**vars(parser.parse()))
