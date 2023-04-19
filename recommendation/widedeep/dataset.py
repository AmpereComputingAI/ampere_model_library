import os
import pathlib
import subprocess


def download_widedeep_processed_data(batch_size):
    from utils.downloads.utils import get_downloads_path

    widedeep_dir_path = pathlib.Path(get_downloads_path(), "widedeep")

    tfrecords_link = "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/widedeep_eval_processed_data.tfrecords"
    tfrecords_file_name = "widedeep_eval_processed_data.tfrecords"

    if not pathlib.Path(widedeep_dir_path).is_dir():
        try:
            subprocess.run(["mkdir", widedeep_dir_path])
            subprocess.run(["wget", tfrecords_link])
            subprocess.run(["mv", tfrecords_file_name, widedeep_dir_path])
        except KeyboardInterrupt:
            subprocess.run(["rm", tfrecords_file_name])
            subprocess.run(["rm", '-rf', widedeep_dir_path])

    os.environ["WIDEDEEP_TFRECORDS_PATH"] = os.path.join(widedeep_dir_path, tfrecords_file_name)

    if batch_size in [1, 2, 4, 8, 16, 32, 50, 64, 100, 128, 200, 256]:
        processed_data_link = "https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/widedeep_processed_data_b" + \
                              str(batch_size)

        if not pathlib.Path(widedeep_dir_path, processed_data_link.split('/')[-1]).is_file():
            try:
                subprocess.run(["wget", processed_data_link])
                subprocess.run(["mv", processed_data_link.split('/')[-1], widedeep_dir_path])
            except KeyboardInterrupt:
                subprocess.run(["rm", processed_data_link.split('/')[-1]])

        os.environ["WIDEDEEP_DATASET_PATH"] = \
            os.path.join(widedeep_dir_path, processed_data_link.split('/')[-1])
