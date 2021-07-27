import os
import csv
import gzip
import json
import pathlib
import numpy as np
import pandas as pd
from datetime import datetime

from collections import Counter
from rich.console import Console
from rich.table import Table

from natural_language_processing.huggingface.profiler_results.utils import get_profiler_results_path

micro_to_minutes = 1.66666667 * 10 ** (-8)
micro_to_milliseconds = 0.001
micro_to_seconds = 0.000001
pd.set_option('display.max_rows', 500)


def calculate_total_time(events_df):
    events_df = events_df.sort_values(by="Start")
    end = -1

    for index, row in events_df.iterrows():

        func_end_time = row["Start"] + row["Duration"]
        if func_end_time > end:
            end = func_end_time

    print(end)
    return end


def variable_read_op_time(events_df):
    return events_df.loc[events_df['Op_Name'] == "ReadVariableOp"]["Duration"].sum()


def print_profiler_results_dls_only(dls_opts_df, total):
    console = Console()
    res_table = Table(show_header=True, header_style="bold magenta", title="TF Profiler results - DLS ops")
    res_table.add_column("DLS Layer", width=15)
    res_table.add_column("Total time ms", width=10)
    res_table.add_column("Ratio of total time %", width=5)
    res_table.add_column("Shape", width=70)
    res_table.add_column("Number of occurrences", width=5)

    for index, row in dls_opts_df.iterrows():
        name = row["Op_Name"].replace(":_DLSSuperNode", '').replace("DLS-SN-", '')
        in_milliseconds = row["Duration"] * micro_to_milliseconds
        ratio = row["Duration"] / total * 100
        res_table.add_row(name, "{:.4f}".format(in_milliseconds), "{:.2f}".format(ratio), row["Shape"],
                          str(row["occurrences"]))
    console.print(res_table)


def print_profiler_results_all(events_df, total):
    console = Console()
    res_table = Table(show_header=True, header_style="bold magenta", title="TF Profiler results - All ops")
    res_table.add_column("DLS Layer", width=45)
    res_table.add_column("Total time ms", width=20)
    res_table.add_column("Ratio of total time %", width=10)
    res_table.add_column("Shape", width=35)
    res_table.add_column("Number of occurrences", width=5)

    for index, row in events_df.iterrows():
        name = row["Op_Name"].replace(":_DLSSuperNode", '').replace("DLS-SN-", '')
        in_milliseconds = row["Duration"] * micro_to_milliseconds
        ratio = row["Duration"] / total * 100
        res_table.add_row(name, "{:.4f}".format(in_milliseconds), "{:.2f}".format(ratio), row["Shape"],
                          str(row["occurrences"]))
    console.print(res_table)


# still under development
# def print_profiler_results_all_by_occurrences(events_df, total):
#     console = Console()
#     res_table = Table(show_header=True, header_style="bold magenta", title="TF Profiler results - All ops")
#     res_table.add_column("Layer", width=45)
#     res_table.add_column("Total time ms", width=20)
#     res_table.add_column("Ratio of total time %", width=10)
#     res_table.add_column("Number of occurrences", width=5)

    # df = events_df.groupby('Op_Name').Duration.sum()
    # df1 = events_df.Op_Name.value_counts()
    # test = pd.concat([df, df1], axis=1).reset_index()
    # df_new = test.rename(columns={'index': 'Op_Name', 'Op_Name': 'Occurrences'})
    # tetete = df_new.sort_values(by=['Occurrences'], ascending=False)
    #
    # for index, row in tetete.iterrows():
    #     name = row["Op_Name"]
    #     duration = row["Duration"] * micro_to_milliseconds
    #     occurrences = row['Occurrences']
    #     res_table.add_row(name, "{:.4f}".format(duration), str(occurrences))

    # print('-' * 85)
    #
    # unique = events_df.Op_Name.unique()
    # unique_operations_duration = {}
    # # create dictionary with durations of specific operations
    # for i in range(unique.shape[0]):
    #     unique_operations_duration[unique[i]] = 0.0
    #
    # # print(unique_operations_duration)
    #
    # events_df = events_df.sort_values(by="Start")
    # end = -1
    #
    # for index, row in events_df.iterrows():

        # print(row["Op_Name"])
        # print(row["Start"])
        # print(row["Duration"])

    #     func_end_time = row["Start"] + row["Duration"]
    #
    #     unique_operations_duration[row["Op_Name"]] += row["Duration"]
    #     if func_end_time > end:
    #         end = func_end_time
    #
    # print(end)
    #
    # console.print(res_table)
    #
    # print(unique_operations_duration)
    # print('sum: ', sum(unique_operations_duration.values()))


def print_total_vs_dls_time(total, dls_time_sum, variable_read_ops):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Total time", width=45, justify="center")
    table.add_column("DLS time", width=45, justify="center")
    table.add_column("DLS to Total ratio %", width=8, justify="right")
    table.add_column("DLS to Total ratio excle varReadOP %", width=11, justify="right")
    ratio = dls_time_sum / total * 100
    ratio_excl_varReadOP = dls_time_sum / (total - variable_read_ops) * 100
    table.add_row("{:.5f}".format(total * micro_to_seconds), "{:.5f}".format(dls_time_sum * micro_to_seconds),
                  "{:.2f}".format(ratio), "{:.2f}".format(ratio_excl_varReadOP))
    console.print(table)


def print_csv_logfile_location(directory):
    console = Console()
    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("CSV log file stored under address:", width=118, justify="center")
    table.add_row(directory)
    console.print(table)


def generate_otput_csv_file(events_df, dls_opts_df, path_to_logs):
    f = open(path_to_logs + '/output_data.csv', 'w')
    writer = csv.writer(f)
    writer.writerow(['name', 'time in milliseconds', 'ratio', 'occurrences'])
    # events_df.to_csv(path_to_logs + '/output_data_df.csv', index = False)
    # TODO MARCEL add business logic
    f.close()


def print_profiler_results(path_to_logs: str):
    for root, dirs, _ in os.walk(path_to_logs + '/plugins/profile/'):
        for d in dirs:
            directory = os.path.join(root, d)

    print_csv_logfile_location(directory)

    my_items = os.listdir(directory + '/')
    matching = [s for s in my_items if ".trace.json.gz" in s]
    json_path = directory + '/' + matching[-1]

    with gzip.open(json_path, "r") as f:
        data = f.read()
        json_data = json.loads(data.decode('utf-8'))

        data_columns = ["Op_Name", "Start", "Duration", "occurrences", "Shape", "Is_DLS_Op"]
        events_df = pd.DataFrame(columns=data_columns)

        # unique_operations = {}

        for i in range(len(json_data["traceEvents"])):
            name = ""
            start = -1.0
            duration = -1.0
            shape = str("")
            is_DLS_Op = False
            occurrences = 0

            # add key, value pair to dictionary
            # if 'name' in json_data["traceEvents"][i] and json_data["traceEvents"][i].get("name") not in unique_operations.keys():
            #     unique_operations[json_data["traceEvents"][i].get("name")] = 0.0

            if 'name' in json_data["traceEvents"][i]:
                tmp_name = json_data["traceEvents"][i].get("name")
                if tmp_name not in events_df["Op_Name"]:
                    name = tmp_name
                    occurrences = 1
                    if json_data["traceEvents"][i]["name"].find("DLS") != -1:
                        is_DLS_Op = True
                else:
                    occurrences_val = events_df.loc[events_df['Op_Name'] == tmp_name]["occurrences"]
                    occurrences_val = occurrences_val + 1
                    events_df.loc[events_df['Op_Name'] == tmp_name]["occurrences"] = occurrences_val

            if 'dur' in json_data["traceEvents"][i].keys():
                duration = json_data["traceEvents"][i].get("dur")

            if 'ts' in json_data["traceEvents"][i].keys():
                start = json_data["traceEvents"][i].get("ts")

            if 'args' in json_data["traceEvents"][i].keys():
                if 'shape' in json_data["traceEvents"][i]['args'].keys():
                    shape_str = json_data["traceEvents"][i]['args'].get("shape")
                    if bool(shape_str.strip().replace('(', '').replace(')', '')):
                        shape = shape_str
            events_df = events_df.append(pd.DataFrame([[name, start, duration, occurrences, shape, is_DLS_Op]],
                                                      columns=data_columns), ignore_index=True)

        events_df = events_df.sort_values(by="Duration", ascending=False)
        total = calculate_total_time(events_df)

        dls_opts_df = events_df[events_df["Is_DLS_Op"] == True]
        dls_opts_df = dls_opts_df.sort_values(by="Duration", ascending=False)
        dls_time_sum = dls_opts_df["Duration"].sum()

        var_read_op = variable_read_op_time(events_df)
        print_total_vs_dls_time(total, dls_time_sum, var_read_op)
        print_profiler_results_dls_only(dls_opts_df, total)
        # print_profiler_results_all(events_df, total)
        generate_otput_csv_file(events_df, dls_opts_df, path_to_logs)

        # print(unique_operations)


profile_path = None


def dls_profiler_enabled():
    if "DLS_PROFILER" in os.environ and os.environ["DLS_PROFILER"] == "1":
        return True
    else:
        return False


def set_profile_path(model_name):
    global profile_path
    profiler_results_dir_path = pathlib.Path(get_profiler_results_path())
    profile_path = os.path.join(profiler_results_dir_path, "{}_{:%Y_%m_%d_%H_%M_%S}".format(model_name, datetime.now()))
    print(profile_path)


def get_profile_path():
    return profile_path


def summarize_tf_profiling(path_to_profiler_results):
    print_profiler_results(path_to_profiler_results)
    print(f"\nTo visualize TF profiler output run locally:\n tensorboard --logdir={get_profile_path()}")
