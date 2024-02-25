import csv
import random
import sys
import math
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split


def main():
    data = {"fp32": {"mem": {}, "perf": {"best_latency": 0.}}, "fp16": {"mem": {}, "perf": {"best_latency": 0.}}}
    for i in range(9):
        data["fp32"]["mem"][2 ** i] = 3 * 2 ** i * 0.6 ** i
        data["fp16"]["mem"][2 ** i] = 2.5 * 2 ** i * 0.6 ** i
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
                    throughput_per_process = float(row[3]) / int(row[1])
                    latency = throughput_per_process / bs
                    data_fp32["best_latency"] = max(data_fp32["best_latency"], latency)
                except ValueError:
                    continue
                throughput_per_process_single = throughput_per_process  # * (1 + math.log(num_processes, 2)/10) # remove
                if bs not in data_fp32.keys():
                    data_fp32[bs] = {num_processes: {num_threads: throughput_per_process},
                                     1: {num_threads: throughput_per_process_single}}
                    data_fp16[bs] = {num_processes: {num_threads: 1.5 * throughput_per_process},
                                     1: {num_threads: 1.5 * throughput_per_process_single}}
                elif num_processes not in data_fp32[bs].keys():
                    data_fp32[bs][num_processes] = {num_threads: throughput_per_process}
                    data_fp32[bs][1][num_threads] = throughput_per_process_single  # remove
                    data_fp16[bs][num_processes] = {num_threads: 1.5 * throughput_per_process}
                    data_fp16[bs][1][num_threads] = 1.5 * throughput_per_process_single  # remove
                else:
                    data_fp32[bs][num_processes][num_threads] = throughput_per_process
                    data_fp32[bs][1][num_threads] = throughput_per_process_single  # remove
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
            thr1 = subset[128 // threads_per_proc][threads_per_proc]
        except KeyError:
            return 0.
        return num_proc * (thr0 - (thr1 - thr0) * (num_proc - 1) / (128 - 1))


def find_best_setting(predictor, precision, available_memory, available_threads, scenario, system):
    mem = predictor.data[precision]["mem"]
    best_perf = 0.
    best_lat = 0.
    best_config = [-1., -1., -1., -1.]
    for bs in mem.keys():
        for threads_per_proc in [2 ** i for i in range(int(math.log(available_threads, 2)) + 1)]:
            num_proc = 1
            while num_proc * threads_per_proc <= available_threads:
                if mem[bs] * num_proc > available_memory:
                    break
                perf = predictor.predict(precision, bs, num_proc, threads_per_proc)
                if perf > best_perf:
                    lat = perf / (num_proc * bs)
                    if not (scenario == 1 and lat <= best_lat and lat <= 0.8 * predictor.data[precision]["perf"][
                        "best_latency"]):
                        best_lat = lat
                        best_perf = perf
                        best_config = [(bs-128)/128, (num_proc-128)/128, (threads_per_proc-128)/128, (best_perf-1e+4)/1e+4]
                num_proc += 1
    return best_config


def prepare_dataset(predictor):
    x = []
    y = []

    for precision in [("fp32", -1), ("fp16", 1)]:
        for mem in [int(2 ** (n / 8)) for n in range(97)]:  # range(8, 4097, 8):
            for n_threads in set([int(2 ** (n / 8)) for n in range(65)]):  # range(1, 2 * 128 + 1):
                for scenario in [-1, 1]:
                    for system in [0]:
                        x.append([system, precision[1], (mem-2048)/2048, (n_threads-128)/128, scenario])
                        y.append(find_best_setting(predictor, precision[0], mem, n_threads, scenario, system))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=42)

    train_set = Data(x_train, y_train)
    test_set = Data(x_test, y_test)

    model = MLP()

    #optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-1)

    #criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()

    # Creating the dataloader
    train_loader = DataLoader(dataset=train_set, batch_size=16, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=16, shuffle=True)

    # Train the model
    total_loss = 0.
    i = 0
    for x, y in test_loader:
        # print(x)
        # print(y)
        y_pred = model(x)
        # print(y_pred)
        # print("------------")
        total_loss += criterion(y_pred, y)
        i += 1
    print(f"val loss = {total_loss / i}")
    for epoch in range(500):
        total_loss = 0.
        i = 0
        for x, y in train_loader:
            #print(x)
            #print(y)
            y_pred = model(x)
            #print(y_pred)
            #print("------------")
            loss = criterion(y_pred, y)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
        print(f"epoch = {epoch}, loss = {total_loss/i}")
        total_loss = 0.
        i = 0
        for x, y in test_loader:
            #print(x)
            #print(y)
            y_pred = model(x)
            #print(y_pred)
            #print("------------")
            total_loss += criterion(y_pred, y)
            i += 1
        print(f"val loss = {total_loss/i}")
    print("Done training!")

    model.eval()
    with torch.no_grad():
        out = model(torch.tensor([0., -1., (60-2048)/2048, (56-128)/128, 1]))
        print(out[0]*128+128, out[1]*128+128, out[2]*128+128, out[3]*1e+4+1e+4)
        x = find_best_setting(predictor, 'fp32', 60, 56, 1, 0)
        print(x[0] * 128 + 128, x[1] * 128 + 128, x[2] * 128 + 128, x[3] * 1e+4 + 1e+4)
        out = model(torch.tensor([0., -1., (60 - 2048) / 2048, (56 - 128) / 128, -1]))
        print(out[0]*128+128, out[1]*128+128, out[2]*128+128, out[3]*1e+4+1e+4)
        x= find_best_setting(predictor, 'fp32', 60, 56, -1, 0)
        print(x[0] * 128 + 128, x[1] * 128 + 128, x[2] * 128 + 128, x[3] * 1e+4 + 1e+4)


class Data(Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).float()
        self.len = len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

    def __len__(self):
        return self.len


class MLP(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = torch.nn.Sequential(
            torch.nn.Linear(5, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    main()
