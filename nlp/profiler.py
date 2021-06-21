import gzip
import json
import os

from collections import Counter


def print_profiler_results(path_to_logs: str):

    for root, dirs, _ in os.walk(path_to_logs + '/plugins/profile/'):
        for d in dirs:
            directory = os.path.join(root, d)

    my_items = os.listdir(directory + '/')
    print(any(".trace.json.gz" in s for s in my_items))

    matching = [s for s in my_items if ".trace.json.gz" in s]

    with gzip.open(directory + '/' + matching[0], "r") as f:
        data = f.read()
        j = json.loads(data.decode('utf-8'))

        microToMinutes = 1.66666667 * 10 ** (-8)
        microToMilliseconds = 0.001
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

        total = 0.0
        print(my_dict)

        # total time of running the model
        print("-" * 25, "TOTAL TIME", '-' * 25, '\n')

        for value in my_dict.values():
            total += value
        print(f'total time in minutes: {total * microToMinutes}')
        print('\n' * 3)

        occurrences_counter = Counter(occurrences)
        # ratio and total time par layer
        print("-" * 25, "RATIOS AND TIME PER LAYER", '-' * 25, '\n')
        for pair in my_dict.items():
            print(f'for {pair[0]} total time in milliseconds is: {pair[1] * microToMilliseconds}')
            print(f'for {pair[0]} ratio to total time is: {pair[1] / total}')
            print(f'{pair[0]} has occurred {occurrences_counter.get(pair[0])} times')
            print('*' * 45)
        print('\n' * 3)

