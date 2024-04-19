# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC

TORCH_JIT_TRACE = False  # otherwise, run torch.compile()


def run_pytorch_fp32(model_name, batch_size, num_runs, timeout, **kwargs):
    from utils.benchmark import run_model
    from utils.misc import print_warning_message
    from utils.pytorch import PyTorchRunnerV2, apply_compile, apply_jit_trace_module
    from utils.speech_recognition.libri_speech_v2 import LibriSpeech
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    processor = WhisperProcessor.from_pretrained(model_name)
    model = WhisperForConditionalGeneration.from_pretrained(model_name, torchscript=TORCH_JIT_TRACE)
    model.eval()
    librispeech = LibriSpeech()
    if TORCH_JIT_TRACE:
        waveform = [librispeech.get_input_array() for _ in range(batch_size)]
        input_features = processor(
            waveform, sampling_rate=LibriSpeech.sampling_rate, return_tensors="pt").input_features
        model = apply_jit_trace_module(model, {"generate": input_features})
        librispeech = LibriSpeech()  # reset
        model = model.generate
    else:
        model = apply_compile(model.generate)

    def single_pass_pytorch(_runner, _librispeech):
        waveform = [_librispeech.get_input_array() for _ in range(batch_size)]
        input_features = processor(
            waveform, sampling_rate=LibriSpeech.sampling_rate, return_tensors="pt").input_features
        predicted_ids = _runner.run(sum([x.shape[0] for x in waveform]), input_features)
        _librispeech.submit_transcription(processor.batch_decode(predicted_ids, skip_special_tokens=True)[0])

    runner = PyTorchRunnerV2(model, throughput_only=True)
    print_warning_message("Sampling rate Whisper operates at is 16,000 Hz, therefore throughput values below can be "
                          "divided by 16,000 to derive 'seconds of processed audio per second'")
    return run_model(single_pass_pytorch, runner, librispeech, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser
    whisper_variants = ["openai/whisper-tiny", "openai/whisper-base", "openai/whisper-small", "openai/whisper-medium",
                        "openai/whisper-large", "openai/whisper-large-v2", "openai/whisper-large-v3"]
    whisper_variants = whisper_variants + [f"{name}.en" for name in whisper_variants[:4]]
    parser = DefaultArgParser(["pytorch"])
    parser.require_model_name(whisper_variants)
    parser.ask_for_batch_size(1)
    run_pytorch_fp32(**vars(parser.parse()))
