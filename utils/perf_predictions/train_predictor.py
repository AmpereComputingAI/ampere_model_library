import csv
import sys
import math


def main():
    data = {"fp32": {"mem": {}, "perf": {}}, "fp16": {"mem": {}, "perf": {}}}
    for i in range(9):
        data["fp32"]["mem"][2**i] = 3*2**i*0.6**i
        data["fp16"]["mem"][2 ** i] = 2.5* 2 ** i*0.6**i
    data_fp32 = data["fp32"]["perf"]
    data_fp16 = data["fp16"]["perf"]
    with open(sys.argv[1], "r") as f:
        reader = csv.reader(f)
        for i, row in enumerate(reader):
            if i == 0:
                assert row == ['batch_size', 'num_processes', 'num_threads', 'throughput_total',
                               'start_timestamp', 'finish_timestamp']
            else:
                bs = int(row[0])
                num_processes = int(row[1])
                num_threads = int(row[2])
                try:
                    throughput_per_process = float(row[3])/int(row[1])
                except ValueError:
                    continue
                throughput_per_process_single = throughput_per_process #* (1 + math.log(num_processes, 2)/10) # remove
                if bs not in data_fp32.keys():
                    data_fp32[bs] = {num_processes: {num_threads: throughput_per_process}, 1: {num_threads: throughput_per_process_single}}
                    data_fp16[bs] = {num_processes: {num_threads: 1.5 * throughput_per_process}, 1: {num_threads: 1.5 * throughput_per_process_single}}
                elif num_processes not in data_fp32[bs].keys():
                    data_fp32[bs][num_processes] = {num_threads: throughput_per_process}
                    data_fp32[bs][1][num_threads] = throughput_per_process_single # remove
                    data_fp16[bs][num_processes] = {num_threads: 1.5 * throughput_per_process}
                    data_fp16[bs][1][num_threads] = 1.5 * throughput_per_process_single  # remove
                else:
                    data_fp32[bs][num_processes][num_threads] = throughput_per_process
                    data_fp32[bs][1][num_threads] = throughput_per_process_single # remove
                    data_fp16[bs][num_processes][num_threads] = 1.5 * throughput_per_process
                    data_fp16[bs][1][num_threads] = 1.5 * throughput_per_process_single  # remove

    pred = Predictor(data)
    prepare_dataset(pred)


class Predictor:
    def __init__(self, data):
        self.data = data

    def predict(self, precision, bs, num_proc, threads_per_proc):
        subset = self.data[precision]["perf"][bs]
        try:
            return num_proc * subset[num_proc][threads_per_proc]
        except KeyError:
            pass
        try:
            thr0 = subset[1][threads_per_proc]
            thr1 = subset[128//threads_per_proc][threads_per_proc]
        except KeyError:
            return 0.
        return num_proc * (thr0 - (thr1 - thr0)*(num_proc - 1) / (128 - 1))


def find_best_setting(predictor, precision, available_memory, available_threads, scenario, system):
    mem = predictor.data[precision]["mem"]
    best_perf = 0.
    best_config = None
    for bs in mem.keys():
        for threads_per_proc in [2**i for i in range(int(math.log(available_threads, 2))+1)]:
            num_proc = 1
            while num_proc * threads_per_proc <= available_threads:
                if mem[bs] * num_proc > available_memory:
                    break
                perf = predictor.predict(precision, bs, num_proc, threads_per_proc)
                if perf > best_perf:
                    best_perf = perf
                    best_config = [bs, num_proc, threads_per_proc, best_perf]
                num_proc += 1
    return best_config


def prepare_dataset(predictor):
    for precision in ["fp32", "fp16"]:
        for mem in range(8, 4097, 8):
            for n_threads in range(1, 2*128+1):
                for scenario in [0, 1]:
                    for system in [0]:
                        # find_best_setting(source_data, precision, mem, n_threads, scenario, system)
                        print(find_best_setting(predictor, precision, 100, 60, scenario, system))
                        df



if __name__ == "__main__":
    main()
