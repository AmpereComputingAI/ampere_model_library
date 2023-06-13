import os
import sys
from typing import Any
import numpy as np
import pickle

def single_pass_pytorch(runner, coco):
    image = np.squeeze(coco.get_input_array().astype("uint8"))
    path =  str(coco.path_to_latest_image).split('/')[-1]
    _ = runner.run(path)

class SamMaskGenerator:
    def __init__(self, predictor, embedding):
        self.predictor = predictor
        self.embedding = embedding

    def __call__(self, filename):
        res = self.embedding[filename]
        for k, v in res.items():
            setattr(self.predictor, k, v)
        x = np.random.randint(0, self.predictor.original_size[0])
        y = np.random.randint(0, self.predictor.original_size[1])
        input_point = np.array([[x, y]])
        input_label = np.array([1])            
        mask, score, logit = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

    def eval(self):
        pass

    def _get_name(self):
        return 'SamMaskGenerator'


def run_pytorch(model_path, batch_size, num_runs, timeout, images_path, anno_path, embd_path):
    sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "segment_anything"))
    from utils.benchmark import run_model
    from utils.pytorch import PyTorchRunnerV2, PyTorchRunner
    from utils.misc import print_warning_message
    from utils.cv.coco import COCODataset
    import inspect
    import torch
    # print(inspect.signature(COCODataset))
    # print(inspect.getfile(COCODataset))
    # print(images_path)
    # print(anno_path)
    from segment_anything import sam_model_registry, SamPredictor
    print(inspect.signature(SamPredictor))
    print(inspect.getfile(SamPredictor))
    model_type = "vit_h"
    device = "cpu"

    with open(embd_path,"rb") as f:
        embedding = pickle.load(f)
    coco = COCODataset( batch_size, "RGB", "COCO_val2014_000000000000", images_path,
                        anno_path, sort_ascending=True, task="segm")
    model_class = sam_model_registry[model_type]
    sam = model_class(checkpoint=model_path)
    sam.to(device=device)
    try:
        torch._C._aio_profiler_print()
        AIO = True
    except AttributeError:        
        AIO = False
    print('AIO = ', AIO)
    sam = torch.compile(sam, backend="aio" if AIO else "inductor")
    predictor = SamPredictor(sam)    
    mask_generator = SamMaskGenerator(predictor, embedding)
    runner = PyTorchRunnerV2(mask_generator)

    for i in range(2):
        image = np.squeeze(coco.get_input_array().astype("uint8"))
        path =  str(coco.path_to_latest_image).split('/')[-1]
        _ = runner.run(path)

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
