import gzip
import json
import os
import csv

from collections import Counter


def print_profiler_results(path_to_logs: str):

    for root, dirs, _ in os.walk(path_to_logs + '/plugins/profile/'):
        for d in dirs:
            directory = os.path.join(root, d)
            print('TTETEETETET')
            print(directory)

    my_items = os.listdir(directory + '/')
    print(any(".trace.json.gz" in s for s in my_items))

    matching = [s for s in my_items if ".trace.json.gz" in s]

    print(directory)
    with gzip.open(directory + '/' + matching[0], "r") as f:
        data = f.read()
        j = json.loads(data.decode('utf-8'))

        micro_to_minutes = 1.66666667 * 10 ** (-8)
        micro_to_milliseconds = 0.001
        micro_to_seconds = 0.000001
        my_dict = {}
        occurrences = []

        for i in range(len(j["traceEvents"])):

            if 'name' in j["traceEvents"][i] and j["traceEvents"][i].get("name") not in my_dict.keys():
                # add key, value pair to dictionary
                my_dict[j["traceEvents"][i].get("name")] = 0.0

            if 'dur' in j["traceEvents"][i].keys():
                my_dict[j["traceEvents"][i].get("name")] += j["traceEvents"][i].get("dur")

            if 'name' in j["traceEvents"][i].keys():
                occurrences.append(j["traceEvents"][i].get("name"))

            else:
                continue

        f = open(path_to_logs + '/output_data.csv', 'w')
        writer = csv.writer(f)

        writer.writerow(['name', 'time in milliseconds', 'ratio', 'occurrences'])

        total = 0.0
        print(my_dict)

        # total time of running the model
        # print(runtimes)
        # sorted_runtimes = sorted(runtimes)
        # print(sorted_runtimes)
        print("-" * 25, "TOTAL TIME", '-' * 25, '\n')

        for value in my_dict.values():
            total += value
        print(f'total time in seconds per inference: {total * micro_to_seconds}')
        print('\n' * 3)

        occurrences_counter = Counter(occurrences)
        # ratio and total time par layer
        print("-" * 25, "RATIOS AND TIME PER LAYER", '-' * 25, '\n')
        for pair in my_dict.items():
            in_milliseconds = pair[1] * micro_to_milliseconds
            ratio = pair[1] / total
            occurrences = occurrences_counter.get(pair[0])
            print(f'for {pair[0]} total time in milliseconds is: {in_milliseconds}')
            print(f'for {pair[0]} ratio to total time is: {pair[1] / total}')
            print(f'{pair[0]} has occurred {occurrences} times')
            print('*' * 45)
            row = [pair[0], in_milliseconds, ratio, occurrences]
            writer.writerow(row)
        print('\n' * 3)

        f.close()

