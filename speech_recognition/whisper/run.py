import os
import sys
import torch


def single_pass_pytorch(runner, librispeech):
    audio = torch.from_numpy(librispeech.get_input_array().astype("float32"))
    transcription = runner.run(audio)
    print(transcription)
    dsf


def run_pytorch(model_name, batch_size, num_runs, timeout):
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "whisper"))
    from utils.benchmark import run_model
    from utils.pytorch import PyTorchRunnerV2
    from utils.speech_recognition.libri_speech_v2 import LibriSpeech
    from speech_recognition.whisper.whisper.whisper import load_model
    from speech_recognition.whisper.whisper.whisper.transcribe import transcribe
    model = load_model(model_name)
    model.eval()

    def transcribe_wrapper(audio):
        return transcribe(model, audio, verbose=None)

    runner = PyTorchRunnerV2(transcribe_wrapper)
    librispeech = LibriSpeech()
    return run_model(single_pass_pytorch, runner, librispeech, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser

    parser = DefaultArgParser(["pytorch"])
    parser.require_model_name()
    run_pytorch(**vars(parser.parse()))
