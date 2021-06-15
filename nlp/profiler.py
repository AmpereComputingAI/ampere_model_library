import gzip
import json
from collections import Counter


with gzip.open("logs/plugins/profile/2021_06_15_15_47_42/49524a0fc1dd.trace.json.gz", "r") as f:
    data = f.read()
    j = json.loads(data.decode('utf-8'))
    print(j.keys())

    print('the events are stored in: ', type(j["traceEvents"]))
    # print(type(j["traceEvents"]))
    print('the data is:', type(j))



    print(len(j["traceEvents"]))
    total_time = 0.0
    matmul_total_time = 0.0
    all_operations = set()
    for i in range(len(j["traceEvents"])):
        if 'name' in j["traceEvents"][i].keys():
            all_operations.add(j["traceEvents"][i].get('name'))
            all_operations_list =

        if 'dur' in j["traceEvents"][i].keys():
            time = j["traceEvents"][i].get("dur")
            total_time += time

        if j["traceEvents"][i].get("name") == "MatMul":
            matmul_total_time += j["traceEvents"][i].get("dur")

    print(f'time in minutes: {total_time * 1.66666667 * 10**(-8)}')
    print(f'MatMul time in minutes: {matmul_total_time * 1.66666667 * 10**(-8)}')
    print(len(all_operations))
    print(all_operations)

