from __future__ import print_function
import torch
print(torch.__version__)
import torchvision
print(torchvision.__version__)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import numpy as np
# check if CUDA is available
train_on_gpu = torch.cuda.is_available()

if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')
    
# You may need to restart your runtime prior to this, to let your installation take effect
# Some basic setup:
# Setup detectron2 logger
import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

# import some common libraries
import numpy as np
import cv2
import random
import os
#from google.colab.patches import cv2_imshow
from datetime import datetime

# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog


from detectron2.data import DatasetCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader


import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch
from detectron2.evaluation import COCOEvaluator, DatasetEvaluators, verify_results
from detectron2.utils.logger import setup_logger

import copy
import logging
from typing import Any, Dict, Tuple
import torch
from fvcore.common.file_io import PathManager

from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.layers import ROIAlign
from detectron2.structures import BoxMode

#from densepose import DatasetMapper, DensePoseCOCOEvaluator, add_densepose_config
import detectron2.data.transforms as T
from detectron2.data import DatasetMapper   # the default mapper

from detectron2.data import detection_utils as utils

from fvcore.transforms.transform import TransformList, Transform, NoOpTransform


def mapper(dataset_dict):
 # 自定义mapper
    dataset_dict = copy.deepcopy(dataset_dict)  # 后面要改变这个dict，所以先复制
    image = utils.read_image(dataset_dict["file_name"], format="BGR")  # 读取图片，numpy array
#     image, transforms = T.apply_transform_gens(
#         [T.Resize((800, 800)), T.RandomContrast(0.1, 3), T.RandomSaturation(0.1, 2), T.RandomRotation(angle=[0, 180]), 
#          T.RandomFlip(prob=0.4, horizontal=False, vertical=True), T.RandomCrop('relative_range', (0.4, 0.6))], image)  # 数组增强
    
#     image, transforms = T.apply_transform_gens(
#         [T.Resize((800, 800)), T.RandomContrast(0.1, 3), T.RandomSaturation(0.1, 2),
#          T.RandomFlip(prob=0.4, horizontal=True, vertical=False), T.RandomCrop('relative_range', (0.4, 0.6))], image)
    image, transforms = T.apply_transform_gens(
        [T.Resize((800, 800)), T.RandomContrast(0.1, 3), T.RandomSaturation(0.1, 2),
         T.RandomFlip(prob=0.4, horizontal=True, vertical=False)], image)
    # 数组增强
   
    dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32")) # 转成Tensor

    annos = [
        utils.transform_instance_annotations(obj, transforms, image.shape[:2])
        for obj in dataset_dict.pop("annotations")
        if obj.get("iscrowd", 0) == 0
    ] # 数据增强要同步标注
    instances = utils.annotations_to_instances(annos, image.shape[:2])  # 将标注转成Instance（Tensor）
    dataset_dict["instances"] = utils.filter_empty_instances(instances)  # 去除空的
    return dataset_dict

class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name):
        #output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        output_folder = os.path.join(cfg.OUTPUT_DIR)
        evaluators = [COCOEvaluator(dataset_name, cfg, True, output_folder)]
        return DatasetEvaluators(evaluators)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, mapper=DatasetMapper(cfg, False))

    @classmethod
    def build_train_loader(cls, cfg):#https://detectron2.readthedocs.io/tutorials/data_loading.html#how-the-existing-dataloader-works
        #return build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, True))
        dataloader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True)) #https://github.com/facebookresearch/detectron2/blob/63f11718c68f1ae951caee157b4e10fae4d7e4be/projects/DensePose/densepose/data/dataset_mapper.py
        #T.Augmentation
        #dataloader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=[]))
        #dataloader = build_detection_train_loader(cfg, mapper=mapper)
        #data_loader = build_detection_train_loader(cfg, mapper=mapper)
        #dataloader = build_detection_train_loader(cfg, mapper=MyDatasetMapper(cfg, is_train=True))
        #dataloader = build_detection_train_loader(cfg, mapper=NewDatasetMapper(cfg, is_train=True))
        #https://detectron2.readthedocs.io/_modules/detectron2/data/detection_utils.html#build_augmentation
        #https://github.com/facebookresearch/detectron2/blob/63f11718c68f1ae951caee157b4e10fae4d7e4be/detectron2/data/transforms/augmentation_impl.py
        return dataloader


if __name__ == "__main__":
    print("Start the main function.")
    extrakeys=["truncation", "occlusion"]       
    BaseFolder='/data/cmpe249-f20/WaymoCOCOMulti/trainvalall/'#'/data/cmpe295-liu/Waymo/WaymoCOCO/'
    train_annotation_json=os.path.join(BaseFolder, "annotations_trainallnew.json")
    train_images=BaseFolder#os.path.join(BaseFolder, "Training")
    val_annotation_json=os.path.join(BaseFolder, "annotations_val50new.json")
    val_images=BaseFolder #os.path.join(BaseFolder, "Validation")

    outputpath="/home/010796032/MyRepo/Detectron2output/"

    
    #traindataset_dicts = load_coco_json(train_annotation_json,train_images, "mywaymo_dataset_train", extrakeys)
    # traindataset_dicts = load_coco_json(train_annotation_json,train_images)#, "mywaymo_dataset_train", extrakeys)
    # valdataset_dicts = load_coco_json(val_annotation_json,val_images)

    print("DatasetCatalog.register.")
    # DatasetCatalog.register("waymococo_Training", load_coco_json(train_annotation_json,train_images))
    # DatasetCatalog.register("waymococo_Validation", load_coco_json(val_annotation_json,val_images))
    
    # for d in ["train", "val"]:
    #     #DatasetCatalog.register("myuav1_" + d, lambda d=d: load_mycoco_json("/data/cmpe295-liu/UAVision/VisDrone2019-DET-" + d + "/annotations.json", "/data/cmpe295-liu/UAVision/VisDrone2019-DET-" + d + "/images", extrakeys))
    #     DatasetCatalog.register("waymococo_" + d, lambda d=d: load_coco_json(BaseFolder + "annotations_"+d+"all.json", BaseFolder))

    print("MetadataCatalog.get.")
    FULL_LABEL_CLASSES = ['vehicle', 'pedestrian', 'sign','cyclist']#['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']#['ignored-regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle','bus',  'motor', 'others']
    for d in ["train", "val"]:
        MetadataCatalog.get("waymococo_" + d).set(thing_classes=FULL_LABEL_CLASSES)
    from detectron2.data.datasets import register_coco_instances
    #'waymococo_val' registered by `register_coco_instances`. Therefore no need to trying to convert it to COCO format
    register_coco_instances("waymococo_train", {}, BaseFolder + "annotations_trainallnew.json", BaseFolder)
    register_coco_instances("waymococo_val", {}, BaseFolder + "annotations_val50new.json", BaseFolder)

    print("Model configuration.")
    cfg = get_cfg()
    cfg.OUTPUT_DIR=outputpath #'./output_waymo' #'./output_x101'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")) #faster_rcnn_R_101_FPN_3x.yaml, faster_rcnn_X_101_32x8d_FPN_3x
    cfg.DATASETS.TRAIN = ("waymococo_train",)
    cfg.DATASETS.TEST = ("waymococo_val",)
    cfg.DATALOADER.NUM_WORKERS = 2#1 #2
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    #cfg.MODEL.WEIGHTS = os.path.join('/home/010796032/PytorchWork', "fasterrcnn_x101_fpn_model_final_68b088.pkl")#using the local 
    #cfg.MODEL.WEIGHTS = os.path.join('/home/010796032/PytorchWork/output', "model_0079999.pth")
    #cfg.MODEL.WEIGHTS = os.path.join('/home/010796032/PytorchWork/output_waymo', "model_0179999.pth")
    cfg.MODEL.WEIGHTS = os.path.join('/home/010796032/MyRepo/Detectron2output/', " model_0879999.pth")#model_0794999 "model_0439999.pth")
    cfg.SOLVER.IMS_PER_BATCH = 2 #4
    cfg.SOLVER.LR_SCHEDULER_NAME='WarmupCosineLR'
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 900000# 140000    # you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 #128 #512#128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(FULL_LABEL_CLASSES) #12  # Kitti has 9 classes (including donot care)

    cfg.TEST.EVAL_PERIOD =10000# 5000
    #cfg.INPUT.ROTATION_ANGLES=[0, 90, 180]

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    # predictor = DefaultPredictor(cfg)
    # evaluator = COCOEvaluator("waymococo_val", cfg, False, output_dir=outputpath)
    # val_loader = build_detection_test_loader(cfg, "waymococo_val")
    # inference_on_dataset(predictor.model, val_loader, evaluator)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    trainer = Trainer(cfg)#DefaultTrainer(cfg) #Trainer(cfg)#DefaultTrainer(cfg) 
    trainer.resume_or_load(resume=True)
    trainer.train()
