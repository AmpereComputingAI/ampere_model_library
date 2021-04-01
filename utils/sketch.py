precision = calc_precision(metrics["true_positives"], metrics["false_positives"])
                    recall = calc_recall(metrics["true_positives"], false_negatives)


if recall is None and precision is None:
    # image does not contain any instances and network didn't make a mistake to say otherwise
    continue
if precision is None:
    # image does contain instances but network didn't see any
    precision = 0.0
if recall is None:
    # image does not contain any instances but network said it does
    recall = 0.0