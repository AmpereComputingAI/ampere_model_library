import os
import sys
import hashlib
import pathlib
import subprocess

from utils.downloads.utils import get_downloads_path


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
    print(f"\nFAIL: {message}")
    sys.exit(1)


def print_warning_message(message):
    """
    A function printing a warning message but not killing the program.
    :param message: str
    """
    print(f"\nCAUTION: {message}")


def download_coco_dataset():
    labels_link = "https://ampereaimodelzoo.s3.amazonaws.com/COCO2014_anno_onspecta.json"
    images_link = "https://ampereaimodelzoo.s3.amazonaws.com/COCO2014_onspecta.tar.gz"
    coco_data = pathlib.Path(get_downloads_path(), "coco")

    if not pathlib.Path(coco_data).is_dir():
        try:
            subprocess.run(["wget", labels_link])
            subprocess.run(["wget", images_link])
            subprocess.run(["mkdir", coco_data])
            subprocess.run(["mv", 'COCO2014_anno_onspecta.json', coco_data])
            subprocess.run(["tar", "-xf", 'COCO2014_onspecta.tar.gz', "-C", coco_data])
            subprocess.run(["rm", 'COCO2014_onspecta.tar.gz'])
        except KeyboardInterrupt:
            subprocess.run(["rm", 'COCO2014_anno_onspecta.json'])
            subprocess.run(["rm", 'COCO2014_onspecta.tar.gz'])
            subprocess.run(["rm", '-rf', coco_data])
    else:
        pass

    dataset = pathlib.Path(coco_data, 'COCO2014_onspecta')
    labels = pathlib.Path(coco_data, 'COCO2014_anno_onspecta.json')

    os.environ["COCO_IMG_PATH"] = str(dataset)
    os.environ["COCO_ANNO_PATH"] = str(labels)


def download_squad_1_1_dataset():

    # dataset_link = 'https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/dev-v1.1.json'
    # squad_data = pathlib.Path(get_downloads_path(), "squad")
    #
    # if not pathlib.Path(squad_data).is_dir():
    #     try:
    #         subprocess.run(["wget", dataset_link])
    #         subprocess.run(["mkdir", squad_data])
    #         subprocess.run(["mv", 'dev-v1.1.json', squad_data])
    #
    #     except KeyboardInterrupt:
    #         subprocess.run(["rm", 'dev-v1.1.json'])
    #         subprocess.run(["rm", '-rf', squad_data])
    # else:
    #     pass
    #
    # dataset = pathlib.Path(squad_data, 'dev-v1.1.json')
    #
    # os.environ["SQUAD_V1_1_PATH"] = str(dataset)

    dataset_link1 = 'https://data.deepai.org/squad1.1.zip'
    squad_data = pathlib.Path(get_downloads_path(), "squad")

    if not pathlib.Path(squad_data).is_dir():
        try:
            subprocess.run(["wget", dataset_link])
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
