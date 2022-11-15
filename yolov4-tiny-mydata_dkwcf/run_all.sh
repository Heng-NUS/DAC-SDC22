#!/bin/sh
# Copyright 2020 Xilinx Inc.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,_
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

####training.........
#working at ./darknet  --->  best weight
#input ----model config ./cfg/voc.data ./cfg/yolov4-tiny.cfg 
#      ---weights  ./yolov4-tiny.weights
#output ./backup/best-weigts
./darknet detector train ./yolov4-tiny-mydata_dkwcf/mydata.data \
                         ./yolov4-tiny-mydata_dkwcf/yolov4-tiny.cfg \
                         ./yolov4-tiny-mydata_dkwcf/train/yolov4-tiny_best.weights -map -gpus 1 -dont_show

./darknet detector map ./yolov4-tiny-mydata_dkwcf/mydata.data \
                       ./yolov4-tiny-mydata_dkwcf/yolov4-tiny.cfg \
                       ./yolov4-tiny-mydata_dkwcf/train/yolov4-tiny_best.weights -letterbox -point 0
                         


#####convert.........
#input ./cfg & .weights
#output ./caffe model ./prototxt .caffemodel
#start vitis-ai-caffe
#bash ~/Documents/DPU-PYNQ/vitis-ai-git/docker_run.sh xilinx/vitis-ai-cpu:1.4.916
#conda activate vitis-ai-caffe
python /opt/vitis_ai/conda/envs/vitis-ai-caffe/bin/convert.py \
        ./yolov4-tiny-mydata_dkwcf/yolov4-tiny.cfg ./yolov4-tiny-mydata_dkwcf/train/yolov4-tiny_best.weights \
        ./yolov4-tiny-mydata_dkwcf/convert/yolov4-tiny.prototxt ./yolov4-tiny-mydata_dkwcf/convert/yolov4-tiny.caffemodel

#####quant...........
vai_q_caffe quantize \
-model ./yolov4-tiny-mydata_dkwcf/convert/yolov4-tiny_quant.prototxt \
-calib_iter 1000 \
-weights ./yolov4-tiny-mydata_dkwcf/convert/yolov4-tiny.caffemodel \
-output_dir ./yolov4-tiny-mydata_dkwcf/quantized/ -method 1
#-sigmoided_layers layer133-conv,layer144-conv,layer155-conv \


#####complie
vai_c_caffe \
--prototxt ./yolov4-tiny-mydata_dkwcf/quantized/deploy.prototxt \
--caffemodel ./yolov4-tiny-mydata_dkwcf/quantized/deploy.caffemodel \
--arch ./ultra96_arch.json \
--output_dir ./yolov4-tiny-mydata_dkwcf/compiled/ \
--net_name yolov4-tiny-mydata-dr2cf \
--options "{'mode':'normal','save_kernel':''}";

#####evaluation------float----
/opt/vitis_ai/conda/envs/vitis-ai-caffe/bin/yolo_detect  \
                                ./yolov4-tiny-mydata_dkwcf/convert/yolov4-tiny.prototxt  \
                                ./yolov4-tiny-mydata_dkwcf/convert/yolov4-tiny.caffemodel  \
                                ./mydata/evltest.txt \
                                -out_file ./yolov4-tiny-mydata_dkwcf/evaluated/caffe_result_fp.txt \
                                -confidence_threshold 0.005 \
                                -classes 12 \
                                -labels "boat, building, car, drone, group, horseride, paraglider, person, riding, truck, wakeboard, whale" \
                                -anchorCnt 3 \
		                -model_type yolov3 \
		                -biases "10,14,  23,27,  37,58,  81,82,  135,169,  344,319"

python  ./yolov4-tiny-mydata_dkwcf/evaluated/evaluation.py 
                -mode detection -detection_use_07_metric True  \
                -gt_file ./mydata/test_gt.txt \
                -result_file ./yolov4-tiny-mydata_dkwcf/evaluated/caffe_result_fp.txt \
                -detection_iou 0.5 -detection_thresh 0.005; 

#####evaluation------quant----
/opt/vitis_ai/conda/envs/vitis-ai-caffe/bin/yolo_detect \
                                ./yolov4-tiny-mydata_dkwcf/quantized/quantize_train_test.prototxt \
                                ./yolov4-tiny-mydata_dkwcf/quantized/quantize_train_test.caffemodel \
                                ./mydata/evltest.txt \
                                -out_file ./yolov4-tiny-mydata_dkwcf/evaluated/caffe_result_quant.txt \
                                -confidence_threshold 0.005 \
                                -classes 12 \
                                -labels "boat, building, car, drone, group, horseride, paraglider, person, riding, truck, wakeboard, whale" \
                                -anchorCnt 3 \
				-model_type yolov3 \
				-biases "10,14,  23,27,  37,58,  81,82,  135,169,  344,319"
