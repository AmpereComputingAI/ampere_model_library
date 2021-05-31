# densenet_169
wget -O classification/densenet_169/densenet_169_tf_fp32.pb https://www.dropbox.com/s/rs3s28o8ml07kyk/densenet_169_tf_fp32.pb
wget -O classification/densenet_169/densenet_169_tf_fp16.pb https://www.dropbox.com/s/1s5pc2ww32tkr9f/densenet_169_tf_fp16.pb
wget -O classification/densenet_169/densenet_169_tflite_int8.tflite https://www.dropbox.com/s/7qsrivpuw3f0n2r/densenet_169_tflite_int8.tflite

# mobilenet_v2
wget -O classification/mobilenet_v2/mobilenet_v2_tf_fp32.pb https://www.dropbox.com/s/jnop89eowak1w6n/mobilenet_v2_tf_fp32.pb
wget -O classification/mobilenet_v2/mobilenet_v2_tf_fp16.pb https://www.dropbox.com/s/ppzx4oz8ne9txeq/mobilenet_v2_fp16.pb
wget -O classification/mobilenet_v2/mobilenet_v2_tflite_int8.tflite https://www.dropbox.com/s/s35x24b04apd9b7/mobilenet_v2_tflite_int8.tflite

# resnet_50_v1.5
wget -O classification/resnet_50_v15/resnet_50_v15_tf_fp32.pb https://www.dropbox.com/s/8pth3hamvsvp5q3/resnet_50_v15_tf_fp32.pb
wget -O classification/resnet_50_v15/resnet_50_v15_tf_fp16.pb https://www.dropbox.com/s/wv1b0bdxehpxcg8/resnet_50_v15_tf_fp16.pb
wget -O classification/resnet_50_v15/resnet_50_v15_tflite_int8.tflite https://www.dropbox.com/s/vit8n91bythao72/resnet_50_v15_tflite_int8.tflite

# ssd_mobilenet_v2
wget -O object_detection/ssd_mobilenet_v2/ssd_mobilenet_v2_tf_fp32.pb https://www.dropbox.com/s/lnaqscsqydzlt1e/ssd_mobilenet_v2_tf_fp32.pb
wget -O object_detection/ssd_mobilenet_v2/ssd_mobilenet_v2_tflite_int8.tflite https://www.dropbox.com/s/hdi9a72uawshp2q/ssd_mobilenet_v2_tflite_int8.tflite

# ssd_inception_v2
wget -O object_detection/ssd_inception_v2/ssd_inception_v2_tf_fp32.pb https://www.dropbox.com/s/jbjgimlrctjgkik/ssd_inception_v2_tf_fp32.pb
wget -O object_detection/ssd_inception_v2/ssd_inception_v2_tf_fp16.pb https://www.dropbox.com/s/lib0gld5tpkudue/ssd_inception_v2_tf_fp16.pb

# yolo_v4_tiny
wget -O object_detection/yolo_v4_tiny/yolo_v4_tiny_tf_fp32.tar.gz https://www.dropbox.com/s/2ogna8d0wqa5war/yolo_v4_tiny_tf_fp32.tar.gz
tar -xvf object_detection/yolo_v4_tiny/yolo_v4_tiny_tf_fp32.tar.gz -C object_detection/yolo_v4_tiny/
rm object_detection/yolo_v4_tiny/yolo_v4_tiny_tf_fp32.tar.gz

# vgg_19
wget -O classification/vgg_19/vgg_19_tf_fp32.tar.gz https://www.dropbox.com/s/6jz4h03qqb05qgg/vgg_19.tar.gz
tar -xvf classification/vgg_19/vgg_19_tf_fp32.tar.gz -C classification/vgg_19/
rm classification/vgg_19/vgg_19_tf_fp32.tar.gz
wget -O classification/vgg_19/vgg_19_tf_fp16.pb https://www.dropbox.com/s/jr8p2stcnth8r7g/vgg_19_frozen_fp16.pb
wget -O classification/vgg_19/vgg_19_tflite_int8.tflite https://www.dropbox.com/s/r74fccbs5p3qgwm/vgg_19_quant.tflite


