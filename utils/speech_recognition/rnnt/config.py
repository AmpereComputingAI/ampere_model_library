# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

config = {
    "input": {
        "normalize": "per_feature",
        "sample_rate": 16000,
        "window_size": 0.02,
        "window_stride": 0.01,
        "window": "hann",
        "features": 80,
        "n_fft": 512,
        "frame_splicing": 3,
        "dither": 0.00001,
        "feat_type": "logfbank",
        "normalize_transcripts": True,
        "trim_silence": True,
        "pad_to": 0,   # TODO
        "max_duration": 16.7,
        "speed_perturbation": True,


        "cutout_rect_regions": 0,
        "cutout_rect_time": 60,
        "cutout_rect_freq": 25,


        "cutout_x_regions": 2,
        "cutout_y_regions": 2,
        "cutout_x_width": 6,
        "cutout_y_width": 6,
    },
    "input_eval": {
        "normalize": "per_feature",
        "sample_rate": 16000,
        "window_size": 0.02,
        "window_stride": 0.01,
        "window": "hann",
        "features": 80,
        "n_fft": 512,
        "frame_splicing": 3,
        "dither": 0.00001,
        "feat_type": "logfbank",
        "normalize_transcripts": True,
        "trim_silence": True,
        "pad_to": 0,
    },
    "rnnt": {
        "rnn_type": "lstm",
        "encoder_n_hidden": 1024,
        "encoder_pre_rnn_layers": 2,
        "encoder_stack_time_factor": 2,
        "encoder_post_rnn_layers": 3,
        "pred_n_hidden": 320,
        "pred_rnn_layers": 2,
        "forget_gate_bias": 1.0,
        "joint_n_hidden": 512,
        "dropout": 0.32,
    },
    "labels": {
        "labels": [" ", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "'", "<BLANK>"]
    }
}
