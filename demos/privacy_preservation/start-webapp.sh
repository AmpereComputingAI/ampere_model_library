#!/bin/bash

pattern='*[[:space:]]Running[[:space:]]on[[:space:]]http'
log="output.log"
server_ip=`curl -s https://ipinfo.io/ip`

cd $DEMO_DIR/

OMP_NUM_THREADS=8 AIO_PROCESS_MODE=0 python -u run.py -m lite-model_movenet_singlepose_lightning_3.tflite -d lite-model_efficientdet_lite2_detection_default_1.tflite -p $FLASK_SERVER_PORT >& $log &

echo -en "\nGetting Webapp URL ."
until grep $pattern $log > /dev/null; do sleep 1; echo -n "."; done
echo -en "\nWebapp URL: "

grep $pattern $log | sed -e 's/* Running on //' -e "s/127.0.0.1/$server_ip/"
echo

cd - &> /dev/null