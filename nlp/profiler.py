import gzip
import json
import os
import csv

from collections import Counter
from rich.console import Console
from rich.table import Table


def print_profiler_results(path_to_logs: str):

    for root, dirs, _ in os.walk(path_to_logs + '/plugins/profile/'):
        for d in dirs:
            directory = os.path.join(root, d)


    my_items = os.listdir(directory + '/')

    matching = [s for s in my_items if ".trace.json.gz" in s]

    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("CSV log file stored under adress:", width = 80, justify = "center")
    table.add_row(directory)
    console.print(table)

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
        for value in my_dict.values():
            total += value

        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("TOTAL TIME", width = 80, justify = "center")
        table.add_row("Total time of execution in seconds")
        table.add_row(str(total * micro_to_seconds))
        console.print(table)
   

        res_table = Table(show_header=True, header_style="bold magenta", title = "Profiler results")
        res_table.add_column("Layer", width = 30)
        res_table.add_column("Total time [ms]", width = 15)
        res_table.add_column("Ratio of total time [%]", width = 15)
        res_table.add_column("Number of occurences", width = 10)
       
        # ratio and total time par layer
        occurrences_counter = Counter(occurrences)
        for pair in my_dict.items():
            in_milliseconds = pair[1] * micro_to_milliseconds
            ratio = pair[1] / total
            occurrences = occurrences_counter.get(pair[0])
            ratio = pair[1] / total
            res_table.add_row(pair[0], "{:.4f}".format(in_milliseconds), "{:.3f}".format(ratio), str(occurrences))
            row = [pair[0], in_milliseconds, ratio, occurrences]
            writer.writerow(row)
        console.print(res_table)    
        print('\n' * 3)

        f.close()

