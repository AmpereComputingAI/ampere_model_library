import os
import sys
import numpy as np


def single_pass_pytorch(runner, coco):
    _ = runner.run(np.squeeze(coco.get_input_array().astype("uint8")))
    # TO-DO: figure out accuracy measuring <- SAM network doesn't predict classes (aka. categories), while COCO
    # requires them for calculation of accuracy <- probably we would need to create a subset of original SAM dataset
    # for pred in output:
    #     coco.submit_mask_prediction(
    #         id_in_batch=0,
    #         bbox=pred["bbox"],
    #         score=pred["stability_score"],
    #         category=None,
    #         mask=pred["segmentation"]
    #     )


def run_pytorch(model_path, batch_size, num_runs, timeout, images_path, anno_path):
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "segment_anything"))
    from utils.benchmark import run_model
    from utils.pytorch import PyTorchRunnerV2
    from utils.misc import print_warning_message
    from utils.cv.coco import COCODataset
    from segment_anything import build_sam, SamAutomaticMaskGenerator
    mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=model_path), output_mode="coco_rle")
    runner = PyTorchRunnerV2(mask_generator.generate)
    coco = COCODataset(
        batch_size, "RGB", "COCO_val2014_000000000000", images_path, anno_path, sort_ascending=True, task="segm")
    coco.summarize_accuracy = lambda: print_warning_message("Accuracy testing unavailable for the SAM model (yet).")
    return run_model(single_pass_pytorch, runner, coco, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser
    parser = DefaultArgParser(["pytorch"])
    parser.require_model_path()
    parser.add_argument("--images_path",
                        type=str,
                        help="path to directory with COCO validation images")
    parser.add_argument("--anno_path",
                        type=str,
                        help="path to file with COCO validation annotations")
    run_pytorch(**vars(parser.parse()))
