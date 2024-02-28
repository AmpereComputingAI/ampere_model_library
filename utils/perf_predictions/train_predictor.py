import json
import sys
import math
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

MEMORY_MARGIN_RATIO = 1.2


def main():
    with open(sys.argv[1], "r") as f:
        data = json.load(f)
    print(predict(data, "fp16", 5, 5, 14))
    print(predict(data, "fp32", 5, 5, 14))
    print(predict(data, "fp32", 19, 8, 7))
    print(predict(data, "fp16", 128, 5, 1))


def interpolate(dictionary: dict, target_key: int):
    if len(dictionary) <= 1:
        print(dictionary)
        print(target_key)
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
        print(dictionary)
        print(target_key)
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
    return mem, throughput
    # subset = self.data[precision]["perf"][bs]
    # try:
    #     return num_proc * subset[num_proc][threads_per_proc]
    # except KeyError:
    #     thr0 = subset[1][threads_per_proc]
    #     thr1 = subset[128 // threads_per_proc][threads_per_proc]
    #     # print(thr0)
    #     # print(thr1)
    #     # print("xxxx")
    #     return num_proc * (thr0 - (thr1 - thr0) * (num_proc - 1) / (128 - 1))


def find_best_setting(predictor, precision, available_memory, available_threads, scenario, system):
    mem = predictor.data[precision]["mem"]
    best_perf = 0.
    best_lat = 0.
    bc = [0, 0, 0, 0]
    best_config = [-1., -1., -1., -1.]
    for bs in mem.keys():
        for threads_per_proc in [2 ** i for i in range(int(math.log(available_threads, 2)) + 1)]:
            num_proc = 1
            while num_proc * threads_per_proc <= available_threads:
                if mem[bs] * num_proc > available_memory:
                    break
                try:
                    perf = predictor.predict(precision, bs, num_proc, threads_per_proc)
                except KeyError:
                    num_proc += 1
                    continue
                tpp = threads_per_proc
                if num_proc == 1 and threads_per_proc < available_threads < threads_per_proc * 2:
                    # print(precision, bs, num_proc, threads_per_proc)
                    # print(perf)
                    # print(predictor.predict(precision, bs, num_proc, threads_per_proc * 2))
                    try:
                        perf_2 = perf + (predictor.predict(precision, bs, num_proc, threads_per_proc * 2) - perf) * (
                                available_threads - threads_per_proc) / threads_per_proc
                        # print(perf_2)
                        # print("-----")
                        if perf_2 > perf:
                            perf = perf_2
                            tpp = available_threads
                    except KeyError:
                        pass
                if perf > best_perf:
                    lat = perf / (num_proc * bs)
                    if not (scenario == 1 and lat <= best_lat and lat <= 0.8 * predictor.data[precision]["perf"][
                        "best_latency"]):
                        best_lat = lat
                        best_perf = perf
                        bc = [bs, num_proc, tpp, best_perf]
                        best_config = [(bs - 128) / 128, (num_proc - 128) / 128, (tpp - 128) / 128,
                                       (best_perf - 1e+4) / 1e+4]
                num_proc += 1
    # print([precision, available_memory, available_threads, scenario])
    # print(bc)
    return best_config


def prepare_dataset(predictor):
    x = []
    y = []

    # find_best_setting(predictor, "fp32", 2, 256, -1, 0)
    # dfs

    for precision in [("fp32", -1), ("fp16", 1)]:
        for mem in [int(2 ** (n / 8)) for n in range(97)]:  # range(8, 4097, 8):
            for n_threads in range(1,
                                   2 * 128 + 1):  # set([int(2 ** (n / 16)) for n in range(129)]):  # range(1, 2 * 128 + 1):
                for scenario in [-1, 1]:
                    for system in [0]:
                        x.append([system, precision[1], (mem - 2048) / 2048, (n_threads - 128) / 128, scenario])
                        y.append(find_best_setting(predictor, precision[0], mem, n_threads, scenario, system))

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.05, random_state=42)

    train_set = Data(x_train, y_train)
    test_set = Data(x_test, y_test)

    model = MLP()

    # optimizer = torch.optim.SGD(model.parameters(), lr=1e-1)
    optimizer = torch.optim.Adagrad(model.parameters(), lr=1e-1)

    # criterion = torch.nn.MSELoss()
    criterion = torch.nn.L1Loss()

    # Creating the dataloader
    train_loader = DataLoader(dataset=train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(dataset=test_set, batch_size=32, shuffle=True)

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
    for epoch in range(10000):
        total_loss = 0.
        i = 0
        for x, y in train_loader:
            # print(x)
            # print(y)
            y_pred = model(x)
            # print(y_pred)
            # print("------------")
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
                # print(x)
                # print(y)
                y_pred = model(x)
                # print(y_pred)
                # print("------------")
                total_loss += criterion(y_pred, y)
                i += 1
            print(f"val loss = {total_loss / i}")
    print("Done training!")

    model.eval()
    with torch.no_grad():
        out = model(torch.tensor([0., -1., (60 - 2048) / 2048, (56 - 128) / 128, 1]))
        print(out[0] * 128 + 128, out[1] * 128 + 128, out[2] * 128 + 128, out[3] * 1e+4 + 1e+4)
        x = find_best_setting(predictor, 'fp32', 60, 56, 1, 0)
        print(x[0] * 128 + 128, x[1] * 128 + 128, x[2] * 128 + 128, x[3] * 1e+4 + 1e+4)

        out = model(torch.tensor([0., -1., (60 - 2048) / 2048, (56 - 128) / 128, -1]))
        print(out[0] * 128 + 128, out[1] * 128 + 128, out[2] * 128 + 128, out[3] * 1e+4 + 1e+4)
        x = find_best_setting(predictor, 'fp32', 60, 56, -1, 0)
        print(x[0] * 128 + 128, x[1] * 128 + 128, x[2] * 128 + 128, x[3] * 1e+4 + 1e+4)

        out = model(torch.tensor([0., -1., (2 - 2048) / 2048, (56 - 128) / 128, -1]))
        print(out[0] * 128 + 128, out[1] * 128 + 128, out[2] * 128 + 128, out[3] * 1e+4 + 1e+4)
        x = find_best_setting(predictor, 'fp32', 2, 56, -1, 0)
        print(x[0] * 128 + 128, x[1] * 128 + 128, x[2] * 128 + 128, x[3] * 1e+4 + 1e+4)

        out = model(torch.tensor([0., -1., (250 - 2048) / 2048, (256 - 128) / 128, -1]))
        print(out[0] * 128 + 128, out[1] * 128 + 128, out[2] * 128 + 128, out[3] * 1e+4 + 1e+4)
        x = find_best_setting(predictor, 'fp32', 250, 256, -1, 0)
        print(x[0] * 128 + 128, x[1] * 128 + 128, x[2] * 128 + 128, x[3] * 1e+4 + 1e+4)

        out = model(torch.tensor([0., -1., (250 - 2048) / 2048, (256 - 128) / 128, 1]))
        print(out[0] * 128 + 128, out[1] * 128 + 128, out[2] * 128 + 128, out[3] * 1e+4 + 1e+4)
        x = find_best_setting(predictor, 'fp32', 250, 256, 1, 0)
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
            torch.nn.Linear(5, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 512),
            torch.nn.ReLU(),
            torch.nn.Linear(512, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 4)
        )

    def forward(self, x):
        return self.layers(x)


if __name__ == "__main__":
    main()
