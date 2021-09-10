import argparse
from utils.imagenet import ImageNet
from utils.tf import TFFrozenModelRunner
from utils.tflite import TFLiteRunner
from utils.benchmark import run_model


def parse_args():
    parser = argparse.ArgumentParser(description="Run MobileNet v2 model.")
    parser.add_argument("-m", "--model_path",
                        type=str, required=True,
                        help="path to the model")
    parser.add_argument("-p", "--precision",
                        type=str, choices=["fp32"], required=True,
                        help="precision of the model provided")
    parser.add_argument("-b", "--batch_size",
                        type=int, default=1,
                        help="batch size to feed the model with")
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

    def run_single_pass(tf_runner, imagenet):
        seq_size = 384
        tf_runner.set_input_tensor("input_ids:0", squad.get_input_ids_array(seq_size))
        tf_runner.set_input_tensor("input_mask:0", squad.get_input_mask_array(seq_size))
        tf_runner.set_input_tensor("segment_ids:0", squad.get_segment_ids_array(seq_size))
        output = tf_runner.run()
        for i in range(batch_size):
            imagenet.submit_predictions(
                i,
                imagenet.extract_top1(output["MobilenetV2/Predictions/Reshape_1:0"][i]),
                imagenet.extract_top5(output["MobilenetV2/Predictions/Reshape_1:0"][i])
            )

    dataset = Squad(batch_size, "RGB", images_path, labels_path,
                       pre_processing="Inception", is1001classes=True)
    runner = TFFrozenModelRunner(model_path, ["logits:0"])

    return run_model(run_single_pass, runner, dataset, batch_size, num_of_runs, timeout)


def main():
    args = parse_args()
    if args.precision == "fp32":
        run_tf_fp32(
            args.model_path, args.batch_size, args.num_runs, args.timeout, args.images_path, args.labels_path
        )
    else:
        assert False, f"Behaviour undefined for precision {args.precision}"


if __name__ == "__main__":
    main()
