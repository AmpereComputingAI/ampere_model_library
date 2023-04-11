import os
import sys


def single_pass_pytorch(runner, coco):
    output = pytorch_runner.run(torch.from_numpy(imagenet.get_input_array(shape)))

    for i in range(batch_size):
        imagenet.submit_predictions(
            i,
            imagenet.extract_top1(output[i]),
            imagenet.extract_top5(output[i])
        )


def run_pytorch(model_path, batch_size, num_runs, timeout):
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "segment_anything"))
    import torch
    from utils.benchmark import run_model
    from utils.pytorch import PyTorchRunnerV2
    from segment_anything import build_sam, SamAutomaticMaskGenerator
    mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=model_path), output_mode="coco_rle")
    runner = PyTorchRunnerV2(torch.compile(mask_generator.generate))

    return run_model(single_pass_pytorch, runner, dataset, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser
    parser = DefaultArgParser(["pytorch"])
    parser.require_model_path()
    run_pytorch(**vars(parser.parse()))
