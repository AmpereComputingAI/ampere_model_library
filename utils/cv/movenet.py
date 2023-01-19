import os
import cv2
import numpy as np
import pandas as pd
import skimage.io as io
import tensorflow as tf
from PIL import Image
from tensorflow.keras.utils import Sequence
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import numpy as np
import pandas as pd
import skimage.io as io
from tqdm import tqdm
from enum import Enum
import numpy as np

# Data Generator Constants
DEFAULT_BATCH_SIZE = 12 #NOTE need to test optimal batch size
NUM_COCO_KEYPOINTS = 17 # Number of joints to detect
NUM_COCO_KP_ATTRBS = 3 # (x,y,v) * 17 keypoints
BBOX_SLACK = 1.3 # before augmentation, increase bbox size to 130%

KP_FILTERING_GT = 4 # Greater than x keypoints


# Data filtering constants
BBOX_MIN_SIZE = 900 # Filter out images smaller than 30x30, TODO tweak


#converting anns to df
def get_meta(coco):
    ids = list(coco.imgs.keys())
    for i, img_id in enumerate(ids):
        img_meta = coco.imgs[img_id]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        img_file_name = img_meta['file_name']
        w = img_meta['width']
        h = img_meta['height']
        url = img_meta['coco_url']
        
        yield img_id, img_file_name, w, h, url, anns

def convert_to_df(coco, data_set):
    images_data = []
    persons_data = []
    gen = get_meta(coco)
    for img_id, img_fname, w, h, url, meta in gen:        
        images_data.append({
            'image_id': int(img_id),
            'src_set_image_id': int(img_id), # repeat id to reference after join
            'coco_url': url,
            'path': data_set + '/' + img_fname,
            'width': int(w),
            'height': int(h)
        })
        for m in meta:
            persons_data.append({
                'ann_id': m['id'],
                'image_id': m['image_id'],
                'is_crowd': m['iscrowd'],
                'bbox': m['bbox'],
                'bbox_area' : m['bbox'][2] * m['bbox'][3],
                'area': m['area'],
                'num_keypoints': m['num_keypoints'],
                'keypoints': m['keypoints'],
                'segmentation': m['segmentation']
            })

    images_df = pd.DataFrame(images_data)
    images_df.set_index('image_id', inplace=True)

    persons_df = pd.DataFrame(persons_data)
    persons_df.set_index('image_id', inplace=True)

    return images_df, persons_df

def get_df(path_to_val_anns):    
    val_coco = COCO(path_to_val_anns) # load annotations for validation set
    images_df, persons_df = convert_to_df(val_coco, 'val2017')
    val_coco_df = pd.merge(images_df, persons_df, right_index=True, left_index=True)

    return val_coco_df
    # ^ Dataframe containing all val and test keypoint annotations



def transform_bbox_square(bbox, slack=1):
    """
        Transforms a bounding box anchored at top left corner of shape () to a square with
        edge length being the larger of the bounding box's height or width.
        Only supports square aspect ratios currently.
        ## Parameters
        bbox : {tuple or ndarray of len 4}
            Given as two points, anchored at top left of image being 0,0: left, upper, right, lower
        slack : {int, float}
            The amount of extra padding that should be applied to the edges of the bounding box after
            transforming
        ##
    """
    x, y, w, h = [i for i in bbox]  # (x,y,w,h) anchored to top left
    center_x = x+w/2
    center_y = y+h/2

    if w >= h:
        new_w = w
        new_h = w
    else:
        new_w = h
        new_h = h

    new_w *= slack  # add slack to bbox
    new_h *= slack  # add slack to bbox
    new_x = center_x - new_w/2
    new_y = center_y - new_h/2
    return (round(new_x), round(new_y), round(new_x+new_w), round(new_y+new_h))

class MovenetDataset:

    def __init__(self, anno_path, images_path, size):        
        df = get_df(anno_path)
        df = df.groupby('src_set_image_id').filter(lambda x: x.shape[0] == 1)
        # 
        df = df.loc[df['is_crowd'] == 0] # drop crowd anns
        df = df.loc[df['num_keypoints'] > KP_FILTERING_GT] # drop anns containing x kps
        df = df.loc[df['bbox_area'] > BBOX_MIN_SIZE]
        df = df.reset_index(drop=True)
        print(df.shape[0])
        dataset = DataGenerator(df=df,base_dir=images_path,
                                input_dim=(size,size),
                                output_dim=(4,4),
                                num_hg_blocks=1, # does not matter
                                shuffle=True,
                                batch_size=1,
                                online_fetch=False,
                                is_eval=True)
        self.dataset = iter(dataset)
        self.available_instances = len(dataset)
        self.movenetcoco = MoveNetCoco()
        self.cocoGt = COCO(anno_path)

        self.image_ids = []
        self.oks_values = [] 

    def submit_keypoint_prediction(self, image_id, keypoints_with_scores, metadata):
        pred = keypoints_with_scores[0,0].copy()
        pred[:,:2] *= metadata['input_dim'][0]
        pred[:,:2] = pred[:,:2].round()
        pred[:,[0,1]] = pred[:,[1,0]]
        image_id, oks = self.movenetcoco(metadata, pred.ravel() )
        self.image_ids.append(image_id)
        self.oks_values.append(oks)

    def summarize_accuracy(self):
        """
        A function summarizing the accuracy achieved on the images obtained with get_input_array() calls on which
        predictions done where supplied with submit_bbox_prediction() function.
        """
        return oks_eval(self.image_ids, self.oks_values, self.cocoGt)

class DataGenerator(Sequence):

    def __init__(self, df, base_dir, input_dim, output_dim, num_hg_blocks, shuffle=False, \
        batch_size=DEFAULT_BATCH_SIZE, online_fetch=False, is_eval=False):

        self.df = df                    # df of the the annotations we want
        self.base_dir = base_dir        # where to read imgs from in collab runtime        
        self.input_dim = input_dim      # model requirement for input image dimensions
        self.output_dim = output_dim    # dimesnions of output heatmap of model
        self.num_hg_blocks = num_hg_blocks
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.is_eval = is_eval
        # If true, images will be loaded from url over network rather than filesystem
        self.online_fetch = online_fetch
        self.on_epoch_end()

    # after each epoch, shuffle indices so data order changes
    def on_epoch_end(self):
        self.indices = np.arange(len(self.df))
        if self.shuffle:
            np.random.shuffle(self.indices)

    # number of batches (not number of examples)
    def __len__(self):
        return int(len(self.df) / self.batch_size)
    #this transforms image 
    def transform_image(self, img, bbox):
        new_bbox = transform_bbox_square(bbox, slack=BBOX_SLACK)
        cropped_img = img.crop(box=new_bbox)
        cropped_width, cropped_height = cropped_img.size
        new_img = cv2.resize(np.array(cropped_img), self.input_dim,
                             interpolation=cv2.INTER_LINEAR)
        return new_img, cropped_width, cropped_height, new_bbox[0], new_bbox[1]

    #this transforms keypoints, both happen differently 
    def transform_label(self, label, cropped_width, cropped_height, anchor_x, anchor_y):
        label = [int(v) for v in label]
        # adjust x/y coords to new resized img
        transformed_label = []
        for x, y, v in zip(*[iter(label)]*NUM_COCO_KP_ATTRBS):
            x = round((x-anchor_x) * self.input_dim[0]/cropped_width)
            y = round((y-anchor_y) * self.input_dim[1]/cropped_height)
            # validate kps, throw away if out of bounds
            # TODO: if kp is thrown away then we must update num_keypoints
            if (x > self.input_dim[0] or x < 0) or (y > self.input_dim[1] or y < 0):
                x, y, v = (0, 0, 0)

            transformed_label.append(x)
            transformed_label.append(y)
            transformed_label.append(v)
        return np.asarray(transformed_label) 
    

    # returns batch at index idx

    """
    Returns a batch from the dataset
    ### Parameters:
    idx : {int-type} Batch number to retrieve
    ### Returns:    
    X : ndarray of shape (batch number, input_dim1, input_dim2, 3)
        This corresponds to a batch of images, normalized from [0,255] to [0,1]
    """
    def __getitem__(self, idx):
        # Initialize Batch:
        X = np.empty((self.batch_size, *self.input_dim, 3))

        metadatas = []
        # get the indices of the requested batch
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        for i, data_index in enumerate(indices):
            ann = self.df.loc[data_index]
            img_path = os.path.join(self.base_dir, ann['path'])

            if self.online_fetch:
                img = Image.fromarray(io.imread(ann['coco_url'])).convert('RGB')  # bottleneck opening from URL
            else:
                # bottleneck opening from file system
                img = Image.open(img_path).convert('RGB')

            transformed_img, cropped_width, cropped_height, anchor_x, anchor_y = self.transform_image(
                img, ann['bbox'])
            #transform label is transforming wrt keypoints
            transformed_label = self.transform_label(
                ann['keypoints'], cropped_width, cropped_height, anchor_x, anchor_y)

            if self.is_eval:
                metadata = {}
                metadata["src_set_image_id"] = ann['src_set_image_id']
                metadata["ann_id"] = ann['ann_id']
                metadata["coco_url"] = ann['coco_url']
                metadata["cropped_width"] = cropped_width
                metadata["cropped_height"] = cropped_height
                metadata["anchor_x"] = anchor_x
                metadata["anchor_y"] = anchor_y
                metadata["input_dim"] = self.input_dim
                metadata["output_dim"] = self.output_dim
                metadata["transformed_label"] = transformed_label #DEBUG
                metadata["ground_truth_keypoints"] = ann['keypoints'] #DEBUG
                metadatas.append(metadata)
            #no need to scale
            normalized_img = transformed_img
            
            X[i, ] = normalized_img  

        if self.is_eval:
            return X, metadatas

        return X

## taking pred of movenet and translate back keypoints(along with person) to original coco image dims
class MoveNetCoco:

    def __call__(self, metadata, untransformed_predictions):
        metadata = self._undo_bounding_box_transformations(metadata, untransformed_predictions)
        oks = self._create_oks_obj(metadata)
        image_id = metadata['src_set_image_id']
        return image_id, oks


    def _undo_x(self, metadata, untransformed_x):
      predicted_x = round(untransformed_x * metadata['cropped_width'] / metadata['input_dim'][0] + metadata['anchor_x'])
      return round(predicted_x)

    """
        Parameters
        ----------
        metadata : object
        should be metadata associated to a single image
        untransformed_y : int
        x coordinate to
    """
    def _undo_y(self, metadata, untransformed_y):
      predicted_y = round(untransformed_y * metadata['cropped_height'] / metadata['input_dim'][1] + metadata['anchor_y'])
      return round(predicted_y)

    """
        Parameters
        ----------
        metadata : object
        should be metadata associated to a single image
        untransformed_predictions : list
        a list of precitions that need to be transformed
        Example:  [1,2,0,1,4,666,32...]
    """
    def _undo_bounding_box_transformations(self, metadata, untransformed_predictions):
        untransformed_predictions = untransformed_predictions.flatten()

        predicted_labels = np.zeros(NUM_COCO_KEYPOINTS * NUM_COCO_KP_ATTRBS)
        list_of_scores = np.zeros(NUM_COCO_KEYPOINTS)

        for i in range(NUM_COCO_KEYPOINTS):
            base = i * NUM_COCO_KP_ATTRBS

            x = untransformed_predictions[base]
            y = untransformed_predictions[base + 1]
            conf = untransformed_predictions[base + 2]

            if conf == 0:
                # this keypoint is not predicted
                x_new, y_new, vis_new = 0, 0, 0
            else:
                x_new = self._undo_x(metadata, x)
                y_new = self._undo_y(metadata, y)
                vis_new = 1
                list_of_scores[i] = conf

            predicted_labels[base]     = x_new
            predicted_labels[base + 1] = y_new
            predicted_labels[base + 2] = vis_new

        metadata['predicted_labels'] = predicted_labels
        metadata['score'] = float(np.mean(list_of_scores))
        return metadata

    def _create_oks_obj(self, metadata):
        #oks is dict like iou, format that coco class is accepting to cal cross-val score
        oks_obj = {}
        oks_obj["image_id"] = int(metadata['src_set_image_id'])
        oks_obj["ann_id"] = int(metadata['ann_id'])
        oks_obj["category_id"] = 1
        oks_obj["keypoints"] = metadata['predicted_labels']
        oks_obj["score"] = float(metadata['score'])
        return oks_obj




def oks_eval(image_ids, list_of_predictions, cocoGt):
    cocoDt=cocoGt.loadRes(list_of_predictions)

    # Convert keypoint predictions to int type
    for i in range(len(list_of_predictions)):
        list_of_predictions[i]["keypoints"] = list_of_predictions[i]["keypoints"].astype('int')

    annType = "keypoints"
    cocoEval = COCOeval(cocoGt,cocoDt,annType)
    cocoEval.params.imgIds = image_ids
    cocoEval.params.catIds = [1] # Person category
    cocoEval.evaluate()
    cocoEval.accumulate()
    print('\nSummary: ')
    cocoEval.summarize()
    stats = cocoEval.stats
    oks = {
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]': stats[0],
        'Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ]': stats[1],
        'Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ]': stats[2],
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]': stats[3],
        'Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]': stats[4],
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ]': stats[5],
        'Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ]': stats[6],
        'Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ]': stats[7],
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ]': stats[8],
        'Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ]': stats[9]
    }
    return oks





