import os
import pathlib
import subprocess

from utils.downloads.utils import get_downloads_path


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

    dataset_link = 'https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/dev-v1.1.json'
    squad_data = pathlib.Path(get_downloads_path(), "squad")

    if not pathlib.Path(squad_data).is_dir():
        try:
            subprocess.run(["wget", dataset_link])
            subprocess.run(["mkdir", squad_data])
            subprocess.run(["mv", 'dev-v1.1.json', squad_data])

        except KeyboardInterrupt:
            subprocess.run(["rm", 'dev-v1.1.json'])
            subprocess.run(["rm", '-rf', squad_data])
    else:
        pass

    dataset = pathlib.Path(squad_data, 'dev-v1.1.json')

    os.environ["SQUAD_V1_1_PATH"] = str(dataset)
