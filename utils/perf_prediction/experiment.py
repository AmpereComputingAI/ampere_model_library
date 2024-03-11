# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC

import json
import random

from utils.perf_prediction.predictor import predict
import urllib.request

with urllib.request.urlopen(
        "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/m128_30%40ampere_pytorch_1.10.0%40resnet_50_v1.5.json") as url:
    look_up_data = json.load(url)

num_proc = [2 ** i for i in range(8)]
num_threads = [2 ** i for i in range(8)]
batch_sizes = [2 ** i for i in range(9)]
cases_done = {}
mem_limit = 500
# results = {np: {nt: {bs for bs in batch_sizes} for nt in num_threads} for np in num_proc}
num_cases = 10

for i in range(num_cases):
    case = (random.choice(batch_sizes), random.choice(num_proc), random.choice(num_threads))
    while case in cases_done.keys() or case[1] * case[2] > 128:
        case = (random.choice(batch_sizes), random.choice(num_proc), random.choice(num_threads))
    try:
        mem, result = predict(look_up_data, "fp32", *case)
        if mem > mem_limit:
            result = 0
    except ValueError:
        result = 0
    cases_done[case] = result
print(cases_done)
print(sorted(cases_done.items(), key=lambda x: x[1]))
