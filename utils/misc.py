# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2022, Ampere Computing LLC

import os
import sys
import hashlib
import pathlib
import subprocess


class UnsupportedPrecisionValueError(ValueError):
    """
    An error being raised when requested precision is not available.
    """
    pass


class ModelPathUnspecified(Exception):
    """
    An Exception class being raised when model name is unspecified.
    """
    pass


class OutOfInstances(Exception):
    """
    An exception class being raised as an error in case of lack of further images to process by the pipeline.
    """
    pass


class FrameworkUnsupportedError(Exception):
    """
    An exception class being raised as an error in case of lack of implemented framework pipeline.
    """
    pass


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
    print(f"\n\033[91mFAIL: {message}\033[0m")
    sys.exit(1)


def print_warning_message(message):
    """
    A function printing a warning message but not killing the program.
    :param message: str
    """
    print(f"\n\033[91mCAUTION: {message}\033[0m")


def advertise_aio(framework_name):
    print(f"\n\033[91mYou are running {framework_name} missing Ampere optimizations.\nConsider using AI-dedicated"
          f" Docker images for increased performance.\nAvailable at:"
          f" https://solutions.amperecomputing.com/solutions/ampere-ai\n\033[0m")


def download_squad_1_1_dataset():
    from utils.downloads.utils import get_downloads_path
    dataset_link1 = 'https://data.deepai.org/squad1.1.zip'
    squad_data = pathlib.Path(get_downloads_path(), "squad")

    if not pathlib.Path(squad_data).is_dir():
        try:
            subprocess.run(["wget", dataset_link1])
            subprocess.run(["mkdir", squad_data])
            subprocess.run(["unzip", 'squad1.1.zip'])
            subprocess.run(["mv", 'dev-v1.1.json', squad_data])
            subprocess.run(["mv", 'train-v1.1.json', squad_data])

        except KeyboardInterrupt:
            subprocess.run(["rm", 'dev-v1.1.json'])
            subprocess.run(["rm", 'train-v1.1.json'])
            subprocess.run(["rm", '-rf', squad_data])
    else:
        pass

    dataset = pathlib.Path(squad_data, 'dev-v1.1.json')

    os.environ["SQUAD_V1_1_PATH"] = str(dataset)


def download_conll_2003_dataset():
    from utils.downloads.utils import get_downloads_path
    dataset_link = 'https://data.deepai.org/conll2003.zip'
    conll_data = pathlib.Path(get_downloads_path(), "conll")

    if not pathlib.Path(conll_data).is_dir():
        subprocess.run(["wget", dataset_link])
        subprocess.run(["mkdir", conll_data])
        subprocess.run(["unzip", 'conll2003.zip', '-d', conll_data])

    os.environ["CONLL2003_PATH"] = str(pathlib.Path(conll_data, 'train.txt'))


def download_ampere_imagenet():
    from utils.downloads.utils import get_downloads_path
    labels_link = "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/ampere_imagenet_substitute_labels.txt"
    images_link = "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/ampere_imagenet_substitute.tar.gz"
    imagenet_data = pathlib.Path(get_downloads_path(), "imagenet")

    if pathlib.Path(imagenet_data).is_dir() and len(os.listdir(imagenet_data)) == 0:
        subprocess.run(["rm", '-rf', imagenet_data])

    if not pathlib.Path(imagenet_data).is_dir():
        try:
            subprocess.run(["wget", labels_link])
            subprocess.run(["wget", images_link])
            subprocess.run(["mkdir", imagenet_data])
            subprocess.run(["mv", 'ampere_imagenet_substitute_labels.txt', imagenet_data])
            subprocess.run(["tar", "-xf", 'ampere_imagenet_substitute.tar.gz', "-C", imagenet_data])
            subprocess.run(["rm", 'ampere_imagenet_substitute.tar.gz'])
        except KeyboardInterrupt:
            subprocess.run(["rm", 'ampere_imagenet_substitute_labels.txt'])
            subprocess.run(["rm", 'ampere_imagenet_substitute.tar.gz'])
            subprocess.run(["rm", '-rf', imagenet_data])
    else:
        pass

    dataset = pathlib.Path(imagenet_data)
    labels = pathlib.Path(imagenet_data, 'ampere_imagenet_substitute_labels.txt')

    os.environ["IMAGENET_IMG_PATH"] = str(dataset)
    os.environ["IMAGENET_LABELS_PATH"] = str(labels)


def download_coco_dataset():
    from utils.downloads.utils import get_downloads_path
    labels_link = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
    images_link = "http://images.cocodataset.org/zips/val2014.zip"
    coco_data = pathlib.Path(get_downloads_path(), "coco")

    if not pathlib.Path(coco_data).is_dir():
        try:
            subprocess.run(["wget", labels_link])
            subprocess.run(["wget", images_link])
            subprocess.run(["mkdir", coco_data])
            subprocess.run(["unzip", 'annotations_trainval2014.zip', '-d', coco_data])
            subprocess.run(["unzip", 'val2014.zip', '-d', coco_data])
        except KeyboardInterrupt:
            subprocess.run(["rm", 'val2014.zip'])
            subprocess.run(["rm", 'annotations_trainval2014.zip'])
    else:
        pass

    dataset = pathlib.Path(coco_data, 'val2014')
    labels = pathlib.Path(coco_data, 'annotations', 'instances_val2014.json')

    if "COCO_IMG_PATH" not in os.environ:
        os.environ["COCO_IMG_PATH"] = str(dataset)

    if "COCO_ANNO_PATH" not in os.environ:
        os.environ["COCO_ANNO_PATH"] = str(labels)
