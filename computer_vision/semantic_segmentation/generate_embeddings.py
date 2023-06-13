import sys
import numpy as np
import os
import pickle
import json
sys.path.append('/home/smamindl/Dls/model_zoo_dev/ampere_model_library/')
from utils.cv.coco import COCODataset
from segment_anything import sam_model_registry, SamPredictor
model_type = "vit_h"
device = "cpu"

batch_size = 1
anno_path = '/home/smamindl/Dls/model_zoo_dev/downloads/coco/COCO2014_anno_onspecta.json'

with open(anno_path, 'r') as f:
    content = json.load(f)
content.keys()


n = 5
image_id = set(x['id'] for x in content['images'][:n])
short = {'info': content['info'],
         'images': content['images'][:n],
         'licenses': content['licenses'],
         'annotations': [x for x in content['annotations'] if x['image_id'] in image_id],
         'categories': content['categories']}

anno_path_short = '/home/smamindl/Dls/model_zoo_dev/downloads/coco/COCO2014_anno_onspecta_short.json'
with open(anno_path_short, 'w') as f:
    json.dump(short, f)

images_path = '/home/smamindl/Dls/model_zoo_dev/downloads/coco/COCO2014_onspecta'
model_path = '/home/smamindl/Dls/model_zoo_dev/downloads/sam_vit_h_4b8939.pth'
coco = COCODataset( batch_size, "RGB", "COCO_val2014_000000000000", images_path,
                    anno_path_short, sort_ascending=True, task="segm")

model_class = sam_model_registry[model_type]
sam = model_class(checkpoint=model_path)
sam.to(device=device)
#sam = torch.compile(sam)
predictor = SamPredictor(sam)  
image = coco.get_input_array()
embd = {}
while image is not None:
    image = np.squeeze(image.astype("uint8"))
    predictor.set_image(image)    
    path = str(coco.path_to_latest_image).split('/')[-1]
    embd[path] = {
            'original_size': predictor.original_size,
            'input_size': predictor.input_size,
            'features': predictor.features,
            'is_image_set': True,
        }
    try:
        image = coco.get_input_array()
    except:
         image = None
         print('Finished the dataset')

embedding_file = 'sam_embeddings.pkl'
with open(embedding_file,"wb") as f:
        pickle.dump(embd,f)
