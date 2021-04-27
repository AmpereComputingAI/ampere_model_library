Mobilenet v2
===============

This folder contains the run code for Mobilenet V2. The architectural design is available to be downloaded [here](https://arxiv.org/abs/1704.04861)

Model name: Intel(R) Xeon(R) Platinum 8124M CPU @ 3.00GHz

eudev


|  number of threads |  precision | batch size  |  top-1 | top-5  | latency | throughput |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1                  | fp32       | 1           | 70%    | 91%    | 11 ms   | 91.89 ips  | 
| 1                  | fp32       | 16          | 70%    | 91%    | 84 ms   | 191.17 ips |
| 18                 | fp32       | 1           | 70%    | 91%    | 5 ms    | 183.65 ips |
| 18                 | fp32       | 16          | 70%    | 91%    | 34 ms   | 467.31 ips |


altra

|  number of threads |  precision | batch size  |  top-1 | top-5  | latency | throughput |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1                  | fp32       | 1           | 70%    | 91%    | 11 ms   | 91.89 ips  | 
| 1                  | fp32       | 16          | 70%    | 91%    | 84 ms   | 191.17 ips |
| 18                 | fp32       | 1           | 70%    | 91%    | 5 ms    | 183.65 ips |
| 18                 | fp32       | 16          | 70%    | 91%    | 34 ms   | 467.31 ips |
| 1                  | int8       | 1           | 70%    | 91%    | 84 ms   | 191.17 ips |
| 1                  | int8       | 16          | 70%    | 91%    | 84 ms   | 191.17 ips |
| 18                 | int8       | 1           | 70%    | 91%    | 84 ms   | 191.17 ips |
| 18                 | int8       | 16          | 70%    | 91%    | 84 ms   | 191.17 ips |




Datasets
-------------

* validation dataset can be downloaded from [here](https://www.dropbox.com/s/eed1so87g199915/ILSVRC2012_img_val.tar)
    
* validation dataset labels can de downloaded from [here](http://dl.caffe.berkeleyvision.org/caffe_ilsvrc12.tar.gz)
  * you can unpack them using command in terminal "tar -xvf ./file"
  * The labels are available in "val.txt" file 

  