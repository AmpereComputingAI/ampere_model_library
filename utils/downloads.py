import os
import pathlib
import subprocess

from downloads.utils import get_downloads_path


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
            subprocess.run(["rm", 'COCO2014_onspecta.tar.gz'])
            subprocess.run(["rm", '-rf', coco_data])
    else:
        pass

    dataset = pathlib.Path(coco_data, 'COCO2014_onspecta')
    labels = pathlib.Path(coco_data, 'COCO2014_anno_onspecta.json')

    os.environ["COCO_IMG_PATH"] = str(dataset)
    os.environ["COCO_ANNO_PATH"] = str(labels)
