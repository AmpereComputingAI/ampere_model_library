import os
import sys
import torch


def run_pytorch(model_name, batch_size, num_runs, timeout):
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "whisper"))
    from utils.benchmark import run_model
    from utils.pytorch import PyTorchRunnerV2
    from utils.speech_recognition.libri_speech_v2 import LibriSpeech
    from speech_recognition.whisper.whisper.whisper import load_model
    from speech_recognition.whisper.whisper.whisper.transcribe import transcribe
    model = load_model(model_name)
    model.eval()

    def single_pass_pytorch(_runner, _librispeech):
        array = _librispeech.get_input_array()
        variable_input_lengths.append(array.shape[0])
        print(variable_input_lengths)
        audio = torch.from_numpy(array.astype("float32"))
        _librispeech.submit_transcription(_runner.run(audio)["text"].lstrip().replace(".", "").upper())

    def transcribe_wrapper(audio):
        return transcribe(model, audio, verbose=None)

    runner = PyTorchRunnerV2(transcribe_wrapper)
    librispeech = LibriSpeech()
    variable_input_lengths = []
    return run_model(single_pass_pytorch, runner, librispeech, batch_size, num_runs, timeout, variable_input_lengths)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser

    parser = DefaultArgParser(["pytorch"])
    parser.require_model_name()
    run_pytorch(**vars(parser.parse()))
