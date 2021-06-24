import os
import sys
import hashlib
import tensorflow as tf
import csv


def get_hash_of_a_file(path_to_file):
    """
    A function calculating md5 hash for a file under the supplied path.

    :param path_to_file: str, path to a file of which a hash should be calculated
    :return: string containing the md5 hash of a file
    """
    hash_md5 = hashlib.md5()
    with open(path_to_file, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def get_env_variable(env_var_name: str, fail_message: str):
    """
    A function checking the value of environment variable and returning given fail message if not set.

    :param env_var_name: str
    :param fail_message: str
    """
    try:
        return os.environ[env_var_name]
    except KeyError:
        print_goodbye_message_and_die(fail_message)


def print_goodbye_message_and_die(message):
    """
    A function printing fail message and making program quit with exit code 1.

    :param message: str
    """
    print(f"\nFAIL: {message}")
    sys.exit(1)


def print_warning_message(message):
    """
    A function printing a warning message but not killing the program.

    :param message: str
    """
    print(f"\nCAUTION: {message}")


def class_names_from_csv(class_map_csv_text):
    """Returns list of class names corresponding to score vector."""
    class_names = []
    with open(class_map_csv_text) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            class_names.append(row['display_name'])

    return class_names


def parse_val_file(labels_path, is1001classes, boundary, audioset):
    """
    A function parsing validation file for ImageNet 2012 validation dataset.

    .txt file consists of 50000 lines each holding data on a single image: its file name and 1 label with class best
    describing image's content

    :param labels_path: str, path to file containing image file names and labels
    :param is1001classes: bool, parameter setting whether the tested model has 1001 classes (+ background) or
    original 1000 classes
    :param boundary: int, index to slash the string in an appropriate labels file
    :param audioset: boolean, specify if the underlying labels file belongs to audioset or not
    :return: list of strings, list of ints
    """
    # single line of labels file for ImageNet dataset looks like this "ILSVRC2012_val_00050000.JPEG 456"
    # single line of labels file for AudioSet dataset looks like this "sound01.wav Music"

    with open(labels_path, 'r') as opened_file:
        lines = opened_file.readlines()

    file_names = list()
    labels = list()

    for line in lines:
        file_name = line[:boundary]
        file_names.append(file_name)
        label = line[boundary + 1:]
        if audioset:
            labels.append(label)
        else:
            label = int(line[boundary:])
            if is1001classes:
                labels.append(label + 1)
            else:
                labels.append(label)

    return file_names, labels


