import json
import sys
import math
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

MEMORY_MARGIN_RATIO = 1.2
SATISFACTORY_LATENCY_RATIO = 0.8


def interpolate(dictionary: dict, target_key: int):
    if len(dictionary) <= 1:
        raise ValueError("Can't interpolate")
    x = {int(k): v for k, v in dictionary.items()}
    sorted_x = sorted(x.keys())
    for i, key in enumerate(sorted_x):
        if key > target_key:
            if i == 0:
                raise ValueError("Can't interpolate")
            return (x[sorted_x[i - 1]] +
                    ((target_key - sorted_x[i - 1]) / (sorted_x[i] - sorted_x[i - 1])) * (
                            x[sorted_x[i]] - x[sorted_x[i - 1]]))
    else:
        raise ValueError("Can't interpolate")


def interpolate_recursively(dictionary: dict, target_keys: list[int]):
    if len(target_keys) == 1:
        try:
            return {int(k): v for k, v in dictionary.items()}[target_keys[0]]
        except KeyError:
            return interpolate(dictionary, target_keys[0])
    x = {int(k): v for k, v in dictionary.items()}
    try:
        return interpolate_recursively(x[target_keys[0]], target_keys[1:])
    except KeyError:
        sorted_x = sorted(x.keys())
        for i, key in enumerate(sorted_x):
            if key > target_keys[0]:
                if i == 0:
                    raise ValueError("Can't interpolate")
                lower_case = interpolate_recursively(x[sorted_x[i - 1]], target_keys[1:])
                upper_case = interpolate_recursively(x[sorted_x[i]], target_keys[1:])
                return (lower_case +
                        ((target_keys[0] - sorted_x[i - 1]) / (sorted_x[i] - sorted_x[i - 1]))
                        * (upper_case - lower_case))
        else:
            raise ValueError("Can't interpolate")


def predict(data, precision, bs, num_proc, threads_per_proc):
    try:
        mem = data["results"][precision]["mem"][str(bs)] * num_proc * MEMORY_MARGIN_RATIO
    except KeyError:
        mem = interpolate(data["results"][precision]["mem"], bs) * num_proc * MEMORY_MARGIN_RATIO
    try:
        throughput = data["results"][precision]["perf"][str(bs)][threads_per_proc][num_proc] * num_proc
    except KeyError:
        d = data["results"][precision]["perf"].copy()
        d.pop("lowest_latency_throughput")
        throughput = interpolate_recursively(d, [bs, threads_per_proc, num_proc]) * num_proc
    return mem / (2 << 9), throughput


def find_best_config(
        source_data: dict, precision: str, available_memory_GiB: int, available_threads: int, optimize_latency: bool):
    best_throughput = 0.
    best_throughput_per_unit = 0.
    best_config = [None, None, None, None, None]
    batch_sizes = [int(bs) for bs in source_data["results"][precision]["perf"].keys()
                   if bs != "lowest_latency_throughput"]
    for bs in range(min(batch_sizes), max(batch_sizes) + 1):
        for threads_per_proc in range(1, available_threads + 1):
            num_proc = 1
            while num_proc * threads_per_proc <= available_threads:
                try:
                    mem, throughput = predict(source_data, precision, bs, num_proc, threads_per_proc)
                except ValueError:
                    num_proc += 1
                    continue
                if mem > available_memory_GiB:
                    break
                if throughput > best_throughput:
                    throughput_per_unit = throughput / (num_proc * bs)
                    if optimize_latency:
                        if (throughput_per_unit > best_throughput_per_unit or
                                throughput_per_unit > SATISFACTORY_LATENCY_RATIO *
                                float(source_data["results"][precision]["perf"]["lowest_latency_throughput"])):
                            best_config = [bs, num_proc, threads_per_proc, throughput, throughput_per_unit, mem]
                            best_throughput = throughput
                            best_throughput_per_unit = throughput_per_unit
                    else:
                        best_config = [bs, num_proc, threads_per_proc, throughput, throughput_per_unit, mem]
                        best_throughput = throughput
                num_proc += 1
    return best_config


def scale(value, max_value):
    if value is None:
        return -1.
    half = max_value / 2
    return (value - half) / half


def prepare_dataset(source_data):
    x = []
    y = []
    best_configs = []
    max_throughput = 0.
    max_throughput_per_unit = 0.
    max_bs = -1
    for precision in source_data["results"].keys():
        max_bs = max(max([int(bs) for bs in source_data["results"][precision]["perf"].keys()
                          if bs != "lowest_latency_throughput"]), max_bs)
    assert max_bs != -1
    precisions = sorted(source_data["results"].keys())
    for i, precision in enumerate(precisions):
        precision_value = (2 * i) / (len(precisions) - 1) - 1
        for mem in tqdm(set([int(2 ** (n / 8))
                        for n in range(int(math.log(source_data["max_mem_per_socket"], 2)) * 8 + 1)])):
            mem_value = scale(mem, source_data["max_mem_per_socket"])
            for n_threads in range(1, source_data["threads_per_socket"] + 1):
                n_threads_value = scale(n_threads, source_data["threads_per_socket"])
                for scenario in [-1., 1.]:
                    x.append([precision_value, mem_value, n_threads_value, scenario])
                    best_config = find_best_config(source_data, precision, mem, n_threads, scenario)
                    best_configs.append(best_config)
                    if best_config[3] is not None:
                        max_throughput = max(best_config[3], max_throughput)
                        max_throughput_per_unit = max(best_config[4], max_throughput_per_unit)
    for best_config in best_configs:
        y.append([
            scale(best_config[0], max_bs),
            scale(best_config[1], source_data["threads_per_socket"]),
            scale(best_config[2], source_data["threads_per_socket"]),
            scale(best_config[3], max_throughput),
            scale(best_config[4], max_throughput_per_unit)
        ])
    for i in range(0, 10000, 100):
        print(x[i], y[i])


def main():
    with open(sys.argv[1], "r") as f:
        data = json.load(f)
    print(find_best_config(data, "fp32", 10, 3, True))

    print(find_best_config(data, "fp32", 30, 16, False))
    print(find_best_config(data, "fp32", 30, 80, False))
    print(find_best_config(data, "fp32", 300, 80, False))
    print(find_best_config(data, "fp32", 170, 59, False))
    import time
    for i in range(1, 81):
        a = time.time()
        print(find_best_config(data, "fp32", 170, i, False))
        print(time.time() - a)
        a = time.time()
        print(find_best_config(data, "fp32", 170, i, True))
        print(time.time() - a)
    gfd
    dataset = prepare_dataset(data)
    # print(predict(data, "fp16", 5, 5, 14))
    # print(predict(data, "fp32", 5, 5, 14))
    # print(predict(data, "fp32", 19, 8, 7))
    # print(predict(data, "fp16", 128, 5, 1))


def train_model(x, y):
    # effort at training MLP for the task of prediction is dropped for now as the look-up predictor is fast enough
    # this setup was promising though
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)

    train_set = Data(x_train, y_train)
    test_set = Data(x_test, y_test)

    model = MLP()

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-1)

    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()

    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)

    # Train the model
    for epoch in range(10000):
        total_loss = 0.
        i = 0
        for x, y in train_loader:
            y_pred = model(x)
            loss = criterion(y_pred, y)
            total_loss += loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            i += 1
        if epoch % 10 == 0:
            print(f"epoch = {epoch}, loss = {total_loss / i}")
            total_loss = 0.
            i = 0
            for x, y in test_loader:
                y_pred = model(x)
                total_loss += criterion(y_pred, y)
                i += 1
            print(f"val loss = {total_loss / i}")
    print("Done training!")


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
            torch.nn.Linear(4, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 2048),
            torch.nn.ReLU(),
            torch.nn.Linear(2048, 1024),
            torch.nn.ReLU(),
            torch.nn.Linear(1024, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 5)
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    main()
