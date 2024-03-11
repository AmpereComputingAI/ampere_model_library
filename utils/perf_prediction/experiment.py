# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC

import json
import random

from utils.perf_prediction.predictor import predict
import urllib.request

with urllib.request.urlopen(
        "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/lookups_aml/m128_30%40ampere_pytorch_1.10.0%40bert_large_mlperf_squad.json") as url:  # noqa
    look_up_data = json.load(url)

num_proc = [2 ** i for i in range(8)]
# num_threads = [2 ** i for i in range(8)]
batch_sizes = [2 ** i for i in range(9)]
cases_done = {}
mem_limit = 500
# results = {np: {nt: {bs for bs in batch_sizes} for nt in num_threads} for np in num_proc}
num_cases = 25

prev_best = None
improvement = None
i = 0
while i < 2 or improvement > 1.:
    for _ in range(num_cases):
        all_done = True
        for bs in batch_sizes:
            if not all_done:
                break
            for np in num_proc:
                if (bs, np, 128 // np) not in cases_done.keys():
                    all_done = False
                    break
        if all_done:
            break
        gg = random.choice(num_proc)
        case = (random.choice(batch_sizes), gg, 128 // gg)
        while case in cases_done.keys() or case[1] * case[2] > 128:
            gg = random.choice(num_proc)
            case = (random.choice(batch_sizes), gg, 128 // gg)
        try:
            mem, result = predict(look_up_data, "fp32", *case)
            if mem > mem_limit:
                result = 0
        except ValueError:
            result = 0
        cases_done[case] = result
    best_result = max(cases_done.values())
    # print(best_result)
    if i > 0:
        improvement = best_result / prev_best
    prev_best = best_result
    print(best_result)
    print(improvement)
    print(batch_sizes)
    print(num_proc)
    bss = []
    npp = []
    ntt = []
    for z in sorted(cases_done.items(), key=lambda x: x[1])[-num_cases:]:  # [int(0.5*len(cases_done)):]:
        bss.append(z[0][0])
        npp.append(z[0][1])
        ntt.append(z[0][2])
    batch_sizes = [bs for bs in batch_sizes if max(bss) >= bs >= min(bss)]
    num_proc = [bs for bs in num_proc if max(npp) >= bs >= min(npp)]
    # num_threads = [bs for bs in num_threads if max(ntt) >= bs >= min(ntt)]
    i += 1

print(i * num_cases)
print(prev_best / 0.9)
# print(cases_done)
