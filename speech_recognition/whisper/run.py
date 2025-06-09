# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
try:
    from utils import misc  # noqa
except ModuleNotFoundError:
    import os
    import sys
    filename = "set_env_variables.sh"
    directory = os.path.realpath(__file__).split("/")[:-1]
    for idx in range(1, len(directory) - 1):
        subdir = "/".join(directory[:-idx])
        if filename in os.listdir(subdir):
            print(f"\nPlease run \033[91m'source {os.path.join(subdir, filename)}'\033[0m first.")
            break
    else:
        print(f"\n\033[91mFAIL: Couldn't find {filename}, are you running this script as part of Ampere Model Library?"
              f"\033[0m")
    sys.exit(1)


def run_pytorch(model_name, num_runs, timeout, use_torch_fp16=False):
    import os
    import sys
    import torch
    batch_size = 1
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "whisper"))
    from utils.benchmark import run_model
    from utils.misc import print_warning_message
    from utils.pytorch import PyTorchRunnerV2
    from utils.speech_recognition.libri_speech_v2 import LibriSpeech
    from speech_recognition.whisper.whisper.whisper import load_model
    from speech_recognition.whisper.whisper.whisper.transcribe import transcribe
    model = load_model(model_name)
    model.eval()
    if use_torch_fp16:
        model = model.half()
        model._encoder.half()
        model._decoder.half()

    def single_pass_pytorch(_runner, _librispeech):
        array = _librispeech.get_input_array()
        audio = torch.from_numpy(array.astype("float32"))
        _librispeech.submit_transcription(
            _runner.run(batch_size * array.shape[0], audio)["text"].lstrip().replace(".", "").upper()
        )

    decode_options = {"fp16": use_torch_fp16}

    def transcribe_wrapper(audio):
        return transcribe(model, audio, no_speech_threshold=1.0, verbose=None, **decode_options)

    runner = PyTorchRunnerV2(transcribe_wrapper, throughput_only=True)
    librispeech = LibriSpeech()
    print_warning_message("Sampling rate Whisper operates at is 16,000 Hz, therefore throughput values below can be "
                          "divided by 16,000 to derive 'seconds of processed audio per second'")
    return run_model(single_pass_pytorch, runner, librispeech, batch_size, num_runs, timeout)


def run_pytorch_fp32(model_name, num_runs, timeout):
    return run_pytorch(model_name, num_runs, timeout, use_torch_fp16=False)


def run_pytorch_fp16(model_name, num_runs, timeout):
    return run_pytorch(model_name, num_runs, timeout, use_torch_fp16=True)


def run_pytorch_cuda(model_name, num_runs, timeout, **kwargs):
    import os
    import sys
    import torch
    batch_size = 1
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "whisper"))
    from utils.benchmark import run_model
    from utils.misc import print_warning_message
    from utils.pytorch import PyTorchRunnerV2
    from utils.speech_recognition.libri_speech_v2 import LibriSpeech
    from speech_recognition.whisper.openai_whisper.whisper import load_model
    from speech_recognition.whisper.openai_whisper.whisper.transcribe import transcribe
    model = load_model(model_name)
    model.eval()

    def single_pass_pytorch(_runner, _librispeech):
        array = _librispeech.get_input_array()
        audio = torch.from_numpy(array.astype("float32"))
        _librispeech.submit_transcription(
            _runner.run(batch_size * array.shape[0], audio)["text"].lstrip().replace(".", "").upper()
        )

    def transcribe_wrapper(audio):
        return transcribe(model, audio, no_speech_threshold=1.0, verbose=None)

    runner = PyTorchRunnerV2(transcribe_wrapper)
    librispeech = LibriSpeech()
    print_warning_message("Sampling rate Whisper operates at is 16,000 Hz, therefore throughput values below can be "
                          "divided by 16,000 to derive 'seconds of processed audio per second'")
    return run_model(single_pass_pytorch, runner, librispeech, batch_size, num_runs, timeout)


if __name__ == "__main__":
    import torch
    from utils.helpers import DefaultArgParser
    whisper_variants = ["tiny", "base", "small", "medium", "large"]
    whisper_variants = whisper_variants + [f"{name}.en" for name in whisper_variants[:4]]
    parser = DefaultArgParser(["pytorch"])
    parser.require_model_name(whisper_variants)
    parser.add_argument("-p", "--precision", type=str, choices=["fp32", "fp16"], required=True)

    args = vars(parser.parse())
    if torch.cuda.is_available():
        run_pytorch_cuda(**args)
    elif args["precision"] == "fp32":
        run_pytorch_fp32(**args)
    elif args["precision"] == "fp16":
        run_pytorch_fp16(**args)
