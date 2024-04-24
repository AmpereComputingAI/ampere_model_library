# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2024, Ampere Computing LLC
try:
    from utils import misc
except ModuleNotFoundError:
    import os
    import sys
    filename = "set_env_variables.sh"
    directory = os.path.realpath(__file__).split("/")[:-1]
    for idx in range(1, len(directory)-1):
        subdir = "/".join(directory[:-idx])
        if filename in os.listdir(subdir):
            print(f"\nPlease run \033[91m'source {os.path.join(subdir, filename)}'\033[0m first.")
            break
    else:
        print(f"\n\033[91mFAIL: Couldn't find {filename}, are you running this script as part of Ampere Model Library?"
              f"\033[0m")
    sys.exit(1)


def run_pytorch(model_path, batch_size, num_runs, timeout, images_path, anno_path):
    import os
    import sys
    import numpy as np
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "segment_anything"))
    from utils.benchmark import run_model
    from utils.pytorch import PyTorchRunnerV2
    from utils.misc import print_warning_message
    from utils.cv.coco import COCODataset
    from segment_anything import build_sam, SamAutomaticMaskGenerator

    def single_pass_pytorch(_runner, _coco):
        _ = _runner.run(batch_size, np.squeeze(_coco.get_input_array().astype("uint8")))
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

    mask_generator = SamAutomaticMaskGenerator(build_sam(checkpoint=model_path), output_mode="coco_rle")
    runner = PyTorchRunnerV2(mask_generator.generate)
    coco = COCODataset(batch_size, "RGB", "COCO_val2014_000000000000", images_path,
                       anno_path, sort_ascending=True, task="segm")
    coco.summarize_accuracy = lambda: print_warning_message("Accuracy testing unavailable for the SAM model (yet).")
    return run_model(single_pass_pytorch, runner, coco, batch_size, num_runs, timeout)


if __name__ == "__main__":
    from utils.helpers import DefaultArgParser

    parser = DefaultArgParser(["pytorch"])
    parser.ask_for_batch_size()
    parser.require_model_path()
    parser.add_argument("--images_path",
                        type=str,
                        help="path to directory with COCO validation images")
    parser.add_argument("--anno_path",
                        type=str,
                        help="path to file with COCO validation annotations")
    run_pytorch(**vars(parser.parse()))
