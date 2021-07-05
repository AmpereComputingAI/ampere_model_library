# densenet_169
wget -O classification/densenet_169/densenet_169_tf_fp32.pb https://www.dropbox.com/s/txqyl9tsrza0l55/densenet_169_tf_fp32.pb
wget -O classification/densenet_169/densenet_169_tf_fp16.pb https://www.dropbox.com/s/kaue3ualwq4qphp/densenet_169_tf_fp16.pb
wget -O classification/densenet_169/densenet_169_tflite_int8.tflite https://www.dropbox.com/s/1nd80f3eq3y5d83/densenet_169_tflite_int8.tflite

# mobilenet_v1
wget -O classification/mobilenet_v1/mobilenet_v1_tf_fp32.pb https://www.dropbox.com/s/eqdm9sloz7o10hd/mobilenet_v1_tf_fp32.pb
wget -O classification/mobilenet_v1/mobilenet_v1_tflite_int8.tflite https://www.dropbox.com/s/yhdxaf9wderav2a/mobilenet_v1_tflite_int8.tflite

# mobilenet_v2
wget -O classification/mobilenet_v2/mobilenet_v2_tf_fp32.pb https://www.dropbox.com/s/thl4v2s6ngspkg3/mobilenet_v2_tf_fp32.pb
wget -O classification/mobilenet_v2/mobilenet_v2_tf_fp16.pb https://www.dropbox.com/s/iqo5xchr8tx8qjt/mobilenet_v2_tf_fp16.pb
wget -O classification/mobilenet_v2/mobilenet_v2_tflite_int8.tflite https://www.dropbox.com/s/euxgo5yficcif9i/mobilenet_v2_tflite_int8.tflite

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

# squeezenet
wget -O classification/squeezenet/squeezenet_tf_fp32.pb https://www.dropbox.com/s/h6y4wazwod6qvnm/squeezenet_tf_fp32.pb
wget -O classification/squeezenet/squeezenet_tflite_int8.tflite https://www.dropbox.com/s/bhxiyv8ixwiujt1/squeezenet_tflite_int8.tflite

# resnet_50_v2
wget -O classification/resnet_50_v2/resnet_50_v2_tf_fp32.tar.gz https://www.dropbox.com/s/qcgzapqj2j3vjg4/resnet50_v2.tar.gz
tar -xvf classification/resnet_50_v2/resnet_50_v2_tf_fp32.tar.gz classification/resnet_50_v2/
rm classification/resnet_50_v2/resnet_50_v2_tf_fp32.tar.gz
wget -O classification/resnet_50_v2/resnet_50_v2_tflite_int8.tflite https://www.dropbox.com/s/igfber3q86yq1bh/resnet_50_v2_quant.tflite

# resnet_101_v2
wget -O classification/resnet_101_v2/resnet_101_v2_tf_fp32.pb https://www.dropbox.com/s/ckojofwnj30ouhn/resnet_v2_101_tf_fp32.pb
wget -O classification/resnet_101_v2/resnet_101_v2_tf_fp16.pb https://www.dropbox.com/s/sq1qz2d39x6vtr8/resnet_101_v2_tf_fp16.pb
wget -O classification/resnet_101_v2/resnet_101_v2_tflite_int8.tflite https://www.dropbox.com/s/n5sw2povn7cmdio/resnet_v2_101_tflite_int8.tflite

#inception v2
wget -O classification/inception_v2/inception_v2_fp32.pb https://www.dropbox.com/s/lwm47zymbu7mcdl/inception_v2_tf_fp32.pb
wget -O classification/inception_v2/inception_v2_fp16.pb  https://www.dropbox.com/s/rwf7wzcczncca1k/inception_v2_tf_fp16.pb
wget -O classification/inception_v2/inception_v2_tflite_int8.tflite https://www.dropbox.com/s/jk3qtrboknjsdkr/inception_v2_tflite_int8.tflite

# inception v3
wget -O classification/inception_v3/inception_v3_tf_fp32.pb https://www.dropbox.com/s/ccfmzojpo3v90bv/inception_v3_tf_fp32.pb
wget -O classification/inception_v3/inception_v3_tflite_int8.tflite https://www.dropbox.com/s/wtq3gix7lhyef6t/inception_v3_tflite_int8.tflite

# inception v4
wget -O classification/inception_v4/inception_v4_tf_fp32.pb https://www.dropbox.com/s/icyu1qk33mrnzpt/inception_v4_tf_fp32.pb
wget -O classification/inception_v4/inception_v4_tf_fp16.pb https://www.dropbox.com/s/tw24objllprx3nj/inception_v4_tf_fp16.pb

# MnasNet_1.0_224 (nasnet mobile)
wget -O classification/nasnet_mobile/mnasnet_tf_fp32.pb https://www.dropbox.com/s/2ja6mlrsartkyg8/mnasnet_tf_fp32.pb
wget -O classification/nasnet_mobile/mnasnet_tf_fp16.pb https://www.dropbox.com/s/9vhsd3qfe5dy4hh/mnasnet_tf_fp16.pb
wget -O classification/nasnet_mobile/mnasnet_tflite_int8.tflite https://www.dropbox.com/s/3yrx7f0egyxoaxt/mnasnet_tflite_int8.tflite

# Nasnet Large
wget -O classification/nasnet_large/nasnet_large_tf_fp32.pb https://www.dropbox.com/s/9g41juu9zmebglc/nasnet_large_tf_fp32.pb
wget -O classification/nasnet_large/nasnet_large_tf_fp16.pb https://www.dropbox.com/s/170ybm6ytpnbwge/nasnet_large_tf_fp16.pb

