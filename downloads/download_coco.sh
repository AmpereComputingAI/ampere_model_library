wget https://ampereaimodelzoo.s3.eu-central-1.amazonaws.com/COCO2014_anno_onspecta.json

export COCO_ANNO_PATH=
echo "dataset: $1";
mv COCO2014_anno_onspecta.json $1