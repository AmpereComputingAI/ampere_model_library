# Segment Anything from Meta AI
Some reading
https://segment-anything.com
https://github.com/facebookresearch/segment-anything

## Downloading model + dataset

Download one of available variants of SAM (more detailed description available at the project's git repo linked above)
> wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth
> wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth
> wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

Download and unzip COCO dataset from https://cocodataset.org/#download

> wget http://images.cocodataset.org/zips/val2014.zip
> wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
> unzip val2014.zip
> unzip annotations_trainval2014.zip

## Run SAM
Follow instructions in root directory of Ampere Model Library to setup your environment.
> AIO_NUM_THREADS=16 python3 run.py -m sam_vit_h_4b8939.pth --images_path=val2014 --anno_path=annotations/instances_val2014.json
