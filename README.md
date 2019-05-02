# mask-faster-rcnn
A pytorch implementation of Mask RCNN detection framework based on 
* [endernewton/tf-faster-rcnn](https://github.com/endernewton/tf-faster-rcnn), developed based on TensorFlow + Numpy
* [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), developed based on Pytorch + Numpy

This project supports single-GPU training of ResNet101-based Mask R-CNN (without FPN support). 
The purpose is to support the experiments in [MAttNet](https://github.com/lichengunc/MAttNet), whose [REFER](https://github.com/lichengunc/refer) dataset is a subset of COCO training portion.
Thus our pre-trained model takes COCO_2014_train_minus_refer_valtest + COCO_2014_valminusminival images for training.

## Prerequisites
+ Python3.6 or higher
+ Pytorch 0.4.0 
+ CUDA 9.0  
+ requirements.txt

## Set up Environments
1. Download Code 
```shell
git clone https://github.com/SunshineJunFu/mask-faster-rcnn.git
```
2. Download refer module
```shell
git clone https://github.com/lichengunc/refer data/refer
```
3. Download coco module 
```shell
git clone https://github.com/cocodataset/cocoapi data/coco
```
4. install packages 
```shell
pip install -r requirements.txt
conda install pytorch=0.4.0 cuda90 -c pytorch
conda install torchvision
install tensorflow-gpu 
```
5. compile refer and coco module 
```shell
cd xxx/data/coco/PythonAPI
make 
cd xxx/data/refer
make
```
6. Compile lib module 
As pointed out by [ruotianluo/pytorch-faster-rcnn](https://github.com/ruotianluo/pytorch-faster-rcnn), choose the right `-arch` to compile the cuda code:

  | GPU model  | Architecture |
  | ------------- | ------------- |
  | TitanX (Maxwell/Pascal) | sm_52 |
  | GTX 960M | sm_50 |
  | GTX 1080 (Ti) | sm_61 |
  | Grid K520 (AWS g2.2xlarge) | sm_30 |
  | Tesla K80 (AWS p2.xlarge) | sm_37 |

Compile the CUDA-based `nms` and `roi_pooling` using following simple commands:
```shell
cd lib
make
```
7. Prepare pretrained model `resnet-101` from [here](https://download.pytorch.org/models/resnet101-5d3b4d8f.pth) and save it as `data/imagenet_weights/res101.pth`.

8. prepare train annotations.
* **coco_minus_refer**: Make the `coco_minus_refer` annotation, which is to be saved as `data/coco/annotations/instances_train_minus_refer_valtest2014.json`
```shell
python tools/make_coco_minus_refer_instances.py
```
9. modify the config files `xxx.yml` in `experiments/cfgs/`, especially for the `DATA_DIR` option.
```
EXP_DIR: res101
TRAIN:
  HAS_RPN: True
  IMS_PER_BATCH: 1
  BBOX_NORMALIZE_TARGETS_PRECOMPUTED: True
  RPN_POSITIVE_OVERLAP: 0.7
  RPN_BATCHSIZE: 256
  PROPOSAL_METHOD: gt
  BG_THRESH_LO: 0.0
  DISPLAY: 20
  BATCH_SIZE: 32
  WEIGHT_DECAY: 0.0001
  DOUBLE_BIAS: False
  SNAPSHOT_PREFIX: res101_mask_rcnn
TEST:
  HAS_RPN: True
POOLING_MODE: crop
DATA_DIR: /data1/fujun/zyr/bs0_Matt/MAttNet/data
```

10. download dataset 
Download the cleaned data and extract them into "data" folder
- http://bvisionweb1.cs.unc.edu/licheng/referit/data/refclef.zip
- http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip
- http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip 
- http://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip 
- Download coco dataset(http://cocodataset.org/#download)
```
$DATA_PATH
│——— coco       
|     └── images
|          └── train2014
|          └── val2014
|          └── test2014
|     └── annotations
├── refcoco
│   ├── instances.json
│   ├── refs(google).p
│   └── refs(unc).p
├── refcoco+
│   ├── instances.json
│   └── refs(unc).p
├── refcocog
│   ├── instances.json
│   └── refs(google).p
└── refclef
   	├── instances.json
  	├── refs(unc).p
	└── refs(berkeley).p
```


### Training 
Run by (`notime` as extra/tag)
```bash
./experiments/scripts/train_mask_rcnn_notime.sh 0 coco_minus_refer res101 notime
```
- Train on COCO 2014 [trainval35k](https://github.com/rbgirshick/py-faster-rcnn/tree/master/models) minus refer_valtest, and test on [minival](https://github.com/rbgirshick/py-faster-rcnn/tree/master/models) (800k/1250k), **35.8** on detection and **30.7** on segmentation (nms).

Checking the training process by calling tensorboard, and check it at `server.cs.unc.edu:port_number`
```bash
tensorboard --logdir tensorboard/res101 --port=port_number
```


### Evaluation
Run by (`notime` as extra/tag)
```bash
./experiments/scripts/test_mask_rcnn_notime.sh 0 refcoco res101 notime
```

Detection Comparison:
- Compared with [Faster R-CNN](https://github.com/ruotianluo/pytorch-faster-rcnn) trained/evaluated on the same images.

| *Detection*|     AP     |    AP50        |    AP75  |
|----------|------------|----------------|----------------|
| Faster R-CNN   |  34.1      |    53.7        |    36.8        |
| Our Mask R-CNN   |  35.8      |    55.3        |    38.6        |


Segmentation Comparison:
- We compare with [Mask R-CNN](https://arxiv.org/abs/1703.06870) implementation. Note this comparison is slightly unfair to ours, due to
* We have fewer (~6,500) training images.
* Our training is single GPU.
* The shorter border length in our model is 600px instead of 800px.

| *Segmentation* |     AP     |    AP50    |    AP75  |
|----------|------------|------------|------------|
| Original Mask R-CNN   |  32.7      |    54.2    |    34.0    |
| Our Mask R-CNN     |  30.7      |    52.3    |    32.4    |

### Pretrained Model
We provide the model we used in [MAttNet](https://github.com/lichengunc/MAttNet) for mask comprehension.
* res101-notime-1250k: [UNC Server](http://bvision.cs.unc.edu/licheng/MattNet/pytorch_mask_rcnn/res101_mask_rcnn_iter_1250k.zip)

Download and put the downloaded `.pth` and `.pkl` files into `output/res101/coco_2014_train_minus_refer_valtest+coco_2014_valminusminival/notime` 

### Demo
- Follow the ipython notebook of `test/check_model.ipynb` to test our model.


