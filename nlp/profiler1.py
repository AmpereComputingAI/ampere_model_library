import gzip
import json
from collections import Counter
import os


def profiler1(path_to_logs: str):

    for root, dirs, _ in os.walk(path_to_logs + '/plugins/profile/'):
        for d in dirs:
            directory = os.path.join(root, d)

    my_items = os.listdir(directory + '/')
    print(any(".trace.json.gz" in s for s in my_items))

    matching = [s for s in my_items if ".trace.json.gz" in s]
    matching[0]

    with gzip.open(directory + '/' + matching[0], "r") as f:
        data = f.read()
        j = json.loads(data.decode('utf-8'))

        total_time = 0.0

        read_variable_op_total_time = 0.0
        matmul_total_time = 0.0
        mul_total_time = 0.0
        bias_add_total_time = 0.0
        truncated_normal_total_time = 0.0
        add_v2_total_time = 0.0
        transpose_total_time = 0.0
        batch_mat_mul_V2_total_time = 0.0
        squared_difference_total_time = 0.0
        softmax_total_time = 0.0
        add_total_time = 0.0
        erf_total_time = 0.0
        sub_total_time = 0.0
        resource_gather_total_time = 0.0
        mean_total_time = 0.0
        assign_add_total_time = 0.0
        assign_variable_op_total_time = 0.0
        tile_total_time = 0.0
        var_handle_op_total_time = 0.0
        assign_total_time = 0.0
        fill_total_time = 0.0
        temporary_variable_total_time = 0.0

        microToMinutes = 1.66666667 * 10 ** (-8)
        microToMilliseconds = 0.001
        all_operations = set()
        my_dict = {}
        occurrences = []

        for i in range(len(j["traceEvents"])):
        #
        #     if j["traceEvents"][i].get("name") not in my_dict.keys():
        #         my_dict[j["traceEvents"][i].get("name")] = 0
        #
        #     if 'name' in j["traceEvents"][i]:
        #         my_dict[j["traceEvents"][i].get("name")] += j["traceEvents"][i].get("dur")
        #     else:
        #         continue
        #
        # total = 0.0
        # for value in my_dict.values():
        #     total += value
        #
        # print(total)


            if 'name' in j["traceEvents"][i].keys():
                all_operations.add(j["traceEvents"][i].get('name'))
                # all_operations_list =

            if 'dur' in j["traceEvents"][i].keys():
                time = j["traceEvents"][i].get("dur")
                total_time += time

            if j["traceEvents"][i].get("name") == "ReadVariableOp":
                read_variable_op_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "MatMul":
                matmul_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "Mul":
                mul_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "BiasAdd":
                bias_add_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "TruncatedNormal":
                truncated_normal_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "AddV2":
                add_v2_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "Transpose":
                transpose_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "BatchMatMulV2":
                batch_mat_mul_V2_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "SquaredDifference":
                squared_difference_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "Softmax":
                softmax_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "Add":
                add_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "Erf":
                erf_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "Sub":
                sub_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "ResourceGather":
                resource_gather_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "Mean":
                mean_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "AssignAdd":
                assign_add_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "AssignVariableOp":
                assign_variable_op_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "Tile":
                tile_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "VarHandleOp":
                var_handle_op_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "Assign":
                assign_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "Fill":
                fill_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

            if j["traceEvents"][i].get("name") == "TemporaryVariable":
                temporary_variable_total_time += j["traceEvents"][i].get("dur")
                occurrences.append(j["traceEvents"][i].get("name"))

        occurrences_counter = Counter(occurrences)

        print(f'time in minutes: {total_time * microToMinutes}')

        print('-' * 45)
        print(f'ReadVariableOp in milliseconds: {read_variable_op_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("ReadVariableOp")}')
        print(f'ratio to whole net: {(read_variable_op_total_time/total_time)}')

        print('-' * 45)
        print(f'MatMul time in milliseconds: {matmul_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("MatMul")}')
        print(f'ratio to whole net: {matmul_total_time / total_time}')

        print('-' * 45)
        print(f'Mul in milliseconds: {mul_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("Mul")}')
        print(f'ratio to whole net: {mul_total_time / total_time}')

        print('-' * 45)
        print(f'BiasAdd in milliseconds: {bias_add_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("BiasAdd")}')
        print(f'ratio to whole net: {bias_add_total_time / total_time}')

        print('-' * 45)
        print(f'TruncatedNormal in milliseconds: {truncated_normal_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("TruncatedNormal")}')
        print(f'ratio to whole net: {truncated_normal_total_time / total_time}')

        print('-' * 45)
        print(f'AddV2 in milliseconds: {add_v2_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("AddV2")}')
        print(f'ratio to whole net: {add_v2_total_time / total_time}')

        print('-' * 45)
        print(f'Transpose in milliseconds: {transpose_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("Transpose")}')
        print(f'ratio to whole net: {transpose_total_time / total_time}')

        print('-' * 45)
        print(f'BatchMatMulV2 in milliseconds: {batch_mat_mul_V2_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("BatchMatMulV2")}')
        print(f'ratio to whole net: {batch_mat_mul_V2_total_time / total_time}')

        print('-' * 45)
        print(f'SquaredDifference in milliseconds: {squared_difference_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("SquaredDifference")}')
        print(f'ratio to whole net: {squared_difference_total_time / total_time}')

        print('-' * 45)
        print(f'Softmax in milliseconds: {softmax_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("Softmax")}')
        print(f'ratio to whole net: {softmax_total_time / total_time}')

        print('-' * 45)
        print(f'Add in milliseconds: {add_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("Add")}')
        print(f'ratio to whole net: {add_total_time / total_time}')

        print('-' * 45)
        print(f'Erf in milliseconds: {erf_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("Erf")}')
        print(f'ratio to whole net: {erf_total_time / total_time}')

        print('-' * 45)
        print(f'Sub in milliseconds: {sub_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("Sub")}')
        print(f'ratio to whole net: {sub_total_time / total_time}')

        print('-' * 45)
        print(f'ResourceGather in milliseconds: {resource_gather_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("ResourceGather")}')
        print(f'ratio to whole net: {resource_gather_total_time / total_time}')

        print('-' * 45)
        print(f'Mean in milliseconds: {mean_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("Mean")}')
        print(f'ratio to whole net: {mean_total_time / total_time}')

        print('-' * 45)
        print(f'AssignAdd in milliseconds: {assign_add_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("AssignAdd")}')
        print(f'ratio to whole net: {assign_add_total_time / total_time}')

        print('-' * 45)
        print(f'AssignVariableOp in milliseconds: {assign_variable_op_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("AssignVariableOp")}')
        print(f'ratio to whole net: {assign_variable_op_total_time / total_time}')

        print('-' * 45)
        print(f'Tile in milliseconds: {tile_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("Tile")}')
        print(f'ratio to whole net: {tile_total_time / total_time}')

        print('-' * 45)
        print(f'VarHandleOp in milliseconds: {var_handle_op_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("VarHandleOp")}')
        print(f'ratio to whole net: {var_handle_op_total_time / total_time}')

        print('-' * 45)
        print(f'Assign in milliseconds: {assign_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("Assign")}')
        print(f'ratio to whole net: {assign_total_time / total_time}')

        print('-' * 45)
        print(f'Fill in milliseconds: {fill_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("Fill")}')
        print(f'ratio to whole net: {fill_total_time / total_time}')

        print('-' * 45)
        print(f'TemporaryVariable in milliseconds: {temporary_variable_total_time * microToMilliseconds}')
        print(f'number of occurrences: {occurrences_counter.get("TemporaryVariable")}')
        print(f'ratio to whole net: {temporary_variable_total_time / total_time}')

        # print(len(all_operations))
        # print(all_operations)

