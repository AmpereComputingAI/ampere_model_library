import os
import sys
import csv
import json
from predictor import test_lookup

SYSTEM = "q80_30"
NUM_THREADS = 80
MODEL = "whisper_medium.en"
FRAMEWORK = "ampere_pytorch_1.10.0"
PRECISIONS = ["fp32", "fp16"]


def process_performance(filename, json_file):
    map = {"lowest_latency_throughput": 0.}
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for i, x in enumerate(reader):
            if i == 0:
                assert x == ['batch_size', 'num_processes', 'num_threads', 'throughput_total',
                             'start_timestamp', 'finish_timestamp']
                continue
            if x[3] == "F":
                continue
            bs, num_proc, num_threads = int(x[0]), int(x[1]), int(x[2])
            if bs not in map.keys():
                map[bs] = {}
            if num_threads not in map[bs].keys():
                map[bs][num_threads] = {}
            throughput_per_proc = float(x[3]) / num_proc
            throughput_per_proc_per_unit = throughput_per_proc / bs
            try:
                map[bs][num_threads][num_proc] = min(map[bs][num_threads][num_proc], throughput_per_proc)
            except KeyError:
                map[bs][num_threads][num_proc] = throughput_per_proc
            map["lowest_latency_throughput"] = max(map["lowest_latency_throughput"], throughput_per_proc_per_unit)
    for prec in PRECISIONS:
        if prec in filename:
            json_file["results"][prec]["perf"] = map
            break
    else:
        assert False
    return json_file


def process_memory(filename, json_file):
    bs_to_mem = {}
    with open(filename, "r") as f:
        reader = csv.reader(f)
        for i, x in enumerate(reader):
            if i == 0:
                assert x == ["batch_size", "num_processes", "num_threads", "memory_MiB"]
                continue
            if x[3] == "F":
                continue
            bs = int(x[0])
            if bs not in bs_to_mem.keys():
                bs_to_mem[bs] = float(x[3]) / int(x[1])
            else:
                bs_to_mem[bs] = max(bs_to_mem[int(x[0])], float(x[3]) / int(x[1]))
    for prec in PRECISIONS:
        if prec in filename:
            json_file["results"][prec]["mem"] = bs_to_mem
            break
    else:
        assert False
    return json_file


def main(csv_results_dir: str):
    files = os.listdir(csv_results_dir)

    for prec in PRECISIONS:
        for filename in files:
            if prec in filename:
                break
        else:
            assert False, f"{prec} data not found in directory {csv_results_dir}"

    json_file = {"model": MODEL, "system": SYSTEM, "num_threads": NUM_THREADS, "framework": FRAMEWORK,
                 "results": {prec: {"perf": None, "mem": None} for prec in PRECISIONS}}
    for f in files:
        assert MODEL in f
        if "memory" not in f:
            json_file = process_performance(os.path.join(csv_results_dir, f), json_file)
        elif "memory" in f:
            json_file = process_memory(os.path.join(csv_results_dir, f), json_file)
        else:
            assert False

    filename = f"{SYSTEM}@{FRAMEWORK}@{MODEL}.json"
    with open(filename, "w") as f:
        json.dump(json_file, f)
    return filename


if __name__ == "__main__":
    test_lookup(main(sys.argv[1]))
