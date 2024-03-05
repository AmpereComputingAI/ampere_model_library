import os
import sys


def run_pytorch_fp32(model_name, num_runs, timeout):
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

    def single_pass_pytorch(_runner, _librispeech):
        array = _librispeech.get_input_array()
        audio = torch.from_numpy(array.astype("float32"))
        _librispeech.submit_transcription(
            _runner.run(batch_size * array.shape[0], audio)["text"].lstrip().replace(".", "").upper()
        )

    def transcribe_wrapper(audio):
        return transcribe(model, audio, no_speech_threshold=1.0, verbose=None)

    runner = PyTorchRunnerV2(transcribe_wrapper, throughput_only=True)
    librispeech = LibriSpeech()
    print_warning_message("Sampling rate Whisper operates at is 16,000 Hz, therefore throughput values below can be "
                          "divided by 16,000 to derive 'seconds of processed audio per second'")
    return run_model(single_pass_pytorch, runner, librispeech, batch_size, num_runs, timeout)


def run_pytorch_cuda(model_name, num_runs, timeout):
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
    from utils.helpers import DefaultArgParser
    whisper_variants = ["tiny", "base", "small", "medium", "large"]
    whisper_variants = whisper_variants + [f"{name}.en" for name in whisper_variants[:4]]
    parser = DefaultArgParser(["pytorch"])
    parser.require_model_name(whisper_variants)

    import torch
    if torch.cuda.is_available():
        run_pytorch_cuda(**vars(parser.parse()))
    else:
        run_pytorch_fp32(**vars(parser.parse()))
