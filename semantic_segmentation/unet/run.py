import os
import argparse
from utils.tf import TFSavedModelRunner
from pathlib import Path
import pickle

DATASET = '/onspecta/dev/mz/temp/datasets/kits19_preprocessed/preprocessed_files.pkl'


def parse_args():
    parser = argparse.ArgumentParser(description="Run Unet model.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=False,
                        help="path to the model")
    parser.add_argument("--timeout",
                        type=float, default=60.0,
                        help="timeout in seconds")
    parser.add_argument("--num_runs",
                        type=int,
                        help="number of passes through network to execute")
    parser.add_argument("--images_path",
                        type=str,
                        help="path to directory with ImageNet validation images")
    parser.add_argument("--labels_path",
                        type=str,
                        help="path to file with validation labels")
    return parser.parse_args()


def run_tf_fp32(model_path, batch_size, num_of_runs, timeout, images_path, labels_path):

    print('loading a model...')
    loaded_model = TFSavedModelRunner(model_path)
    model = loaded_model.signatures["serving_default"]

    with open(Path(DATASET), "rb") as f:
        preprocess_files = pickle.load(f)['file_list']

    count = len(preprocess_files)
    print(count)
    print(type(preprocess_files))
    print(preprocess_files)

    def do_infer(self, input_tensor):
        """
        Perform inference upon input_tensor with TensorFlow
        """
        test = model(input_tensor)
        return test



    def load_query_samples(self, sample_list):
        """
        Opens preprocessed files (or query samples) and loads them into memory
        """
        for sample_id in sample_list:
            file_name = self.preprocess_files[sample_id]
            print("Loading file {:}".format(file_name))
            with open(Path(self.preprocessed_data_dir, "{:}.pkl".format(file_name)), "rb") as f:
                self.loaded_files[sample_id] = pickle.load(f)[0]


def main():
    args = parse_args()
    run_tf_fp32(
        args.model_path, 1, args.num_runs, args.timeout, args.images_path, args.labels_path
    )


if __name__ == "__main__":
    main()
