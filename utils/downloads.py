import subprocess


def download_coco_dataset():
    labels = "https://ampereaimodelzoo.s3.amazonaws.com/COCO2014_anno_onspecta.json"
    images = "https://ampereaimodelzoo.s3.amazonaws.com/COCO2014_onspecta.tar.gz"
    coco_data = pathlib.Path(get_downloads_path(), "coco")

    if not pathlib.Path(coco_data).is_dir():
        try:
            subprocess.run(["wget", labels])
            subprocess.run(["wget", images])
            subprocess.run(["mkdir", coco_data])
            subprocess.run(["mv", 'COCO2014_anno_onspecta.json', coco_data])
            subprocess.run(["tar", "-xf", 'COCO2014_onspecta.tar.gz', "-C", coco_data])
            subprocess.run(["rm", 'COCO2014_onspecta.tar.gz'])
        except KeyboardInterrupt:
            subprocess.run(["rm", 'COCO2014_onspecta.tar.gz'])
            subprocess.run(["rm", '-rf', coco_data])
