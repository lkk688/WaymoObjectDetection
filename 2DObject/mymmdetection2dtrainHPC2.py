# Check Pytorch installation
import torch, torchvision
print(torch.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

from mmdet.apis import inference_detector, init_detector, show_result_pyplot

# Choose to use a config and initialize the detector
Basemmdetection='/home/010796032/3DObject/mmdetection/'
config = Basemmdetection+'configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py'
# Setup a checkpoint file to load
checkpoint = Basemmdetection+'checkpoints/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'
#checkpoint = '/home/014562561/model3/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'
# initialize the detector
#model1 = init_detector(config, checkpoint, device='cuda:0')

data_root = '/data/cmpe249-f20/WaymoCOCOMulti/trainvalall/'
classes=('vehicle', 'pedestrian', 'sign', 'cyclist')
workdir = Basemmdetection+"waymococo_fasterrcnnr101train"

from mmcv import Config
from mmdet.apis import set_random_seed
# cfg = Config.fromfile('/home/014562561/mmdetection/configs/faster_rcnn/faster_rcnn_r50_caffe_fpn_mstrain_1x_coco.py')
cfg = Config.fromfile(config)

# Modify dataset type and path
cfg.dataset_type = 'CocoDataset'
# cfg.data_root = '/content/drive/MyDrive/deepdrive/images/bdd100k/images/100k/'

cfg.data.test.type = 'CocoDataset'
cfg.data.test.data_root = data_root 
cfg.data.test.ann_file = data_root + 'annotations_val20new.json'#'annotations_valallnew.json' 
cfg.data.test.img_prefix =  ''

cfg.data.train.type = 'CocoDataset'
cfg.data.train.data_root = data_root 
cfg.data.train.ann_file = data_root + 'annotations_train684step8allobject.json'#'annotations_trainallnew.json'#'annotations_train200new.json'
cfg.data.train.img_prefix = ''

cfg.data.val.type = 'CocoDataset'
cfg.data.val.data_root = data_root
cfg.data.val.ann_file = data_root + 'annotations_val50new.json'
cfg.data.val.img_prefix = ''


cfg.data.samples_per_gpu = 4 #batch size
cfg.data.workers_per_gpu = 4

# modify num classes of the model in box head
cfg.model.roi_head.bbox_head.num_classes = len(classes)# 10
# We can still use the pre-trained Mask RCNN model though we do not need to
# use the mask branch
cfg.load_from = checkpoint #
cfg.resume_from = workdir + "/epoch_34.pth"

# Set up working dir to save files and logs.
cfg.work_dir = workdir #

# The original learning rate (LR) is set for 8-GPU training.
# We divide it by 8 since we only use one GPU.
cfg.optimizer.lr = 0.02 / 8
cfg.lr_config.warmup = None
cfg.log_config.interval = 10

# Change the evaluation metric since we use customized dataset.
cfg.evaluation.metric = 'bbox'
# We can set the evaluation interval to reduce the evaluation times
cfg.evaluation.interval = 2
# We can set the checkpoint saving interval to reduce the storage cost
cfg.checkpoint_config.interval = 1

# Set seed thus the results are more reproducible
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)
cfg.total_epochs = 35 #24

#cfg.model.pretrained = '~/.cache/torch/hub/checkpoints/resnet50_msra-5891d200.pth'#'/home/014562561/model3/resnet50_msra-5891d200.pth'

# We can initialize the logger for training and have a look
# at the final config used for training
print(f'Config:\n{cfg.pretty_text}')

import copy
import os.path as osp

import mmcv
import numpy as np

from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector

# Build dataset
datasets = build_dataset(cfg.data.train)

# Build the detector
model = build_detector(
    cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
# Add an attribute for visualization convenience
model.CLASSES = classes #("person","car","bus","rider", "truck","bike","motor","train","traffic light","traffic sign")

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_detector(model, datasets, cfg, distributed=False, validate=True)
