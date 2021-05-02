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



if __name__ == "__main__":
    print("Start the main function.")
    extrakeys=["truncation", "occlusion"]       
    BaseFolder='/data/cmpe249-f20/WaymoCOCOMulti/trainvalall/'#'/data/cmpe295-liu/Waymo/WaymoCOCO/'
    train_annotation_json=os.path.join(BaseFolder, "3classsub_annotations_trainall.json")
    train_images=BaseFolder#os.path.join(BaseFolder, "Training")
    val_annotation_json=os.path.join(BaseFolder, "3classsub_annotations_valall.json")
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
    #     DatasetCatalog.register("waymococo2_" + d, lambda d=d: load_coco_json(BaseFolder + "annotations_"+d+"all.json", BaseFolder))
    from detectron2.data.datasets import register_coco_instances
    #'waymococo_val' registered by `register_coco_instances`. Therefore no need to trying to convert it to COCO format
    register_coco_instances("waymococo2_train", {}, BaseFolder + "3classsub_annotations_trainall.json", BaseFolder)
    register_coco_instances("waymococo2_val", {}, BaseFolder + "3classsub_annotations_valall.json", BaseFolder)
    
    print("MetadataCatalog.get.")
    FULL_LABEL_CLASSES = ['vehicle', 'pedestrian', 'sign', 'cyclist']#['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist'] #['vehicle', 'pedestrian', 'sign', 'cyclist']#['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']#['ignored-regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle','bus',  'motor', 'others']
    for d in ["train", "val"]:
        MetadataCatalog.get("waymococo2_" + d).set(thing_classes=FULL_LABEL_CLASSES)
    

    print("Model configuration.")
    cfg = get_cfg()
    cfg.OUTPUT_DIR=outputpath #'./output_waymo' #'./output_x101'
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")) #faster_rcnn_R_101_FPN_3x.yaml, faster_rcnn_X_101_32x8d_FPN_3x
    cfg.DATASETS.TRAIN = ("waymococo2_train",)
    cfg.DATASETS.TEST = ("waymococo2_val",)
    cfg.DATALOADER.NUM_WORKERS = 8 #1 #2
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    #cfg.MODEL.WEIGHTS = os.path.join('/home/010796032/PytorchWork', "fasterrcnn_x101_fpn_model_final_68b088.pkl")#using the local 
    #cfg.MODEL.WEIGHTS = os.path.join('/home/010796032/PytorchWork/output', "model_0079999.pth")
    #cfg.MODEL.WEIGHTS = os.path.join('/home/010796032/PytorchWork/output_waymo', "model_0179999.pth")
    cfg.MODEL.WEIGHTS = os.path.join('/home/010796032/MyRepo/Detectron2output/', "model_0029999.pth")

    cfg.SOLVER.IMS_PER_BATCH = 4
    cfg.SOLVER.LR_SCHEDULER_NAME='WarmupCosineLR'
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LR
    cfg.SOLVER.MAX_ITER = 180000# 140000    # you may need to train longer for a practical dataset
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512 #128 #512#128   # faster, and good enough for this toy dataset (default: 512)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(FULL_LABEL_CLASSES) #12  # Kitti has 9 classes (including donot care)

    cfg.TEST.EVAL_PERIOD =20000# 5000
    #cfg.INPUT.ROTATION_ANGLES=[0, 90, 180]

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print("Current Time =", current_time)

    predictor = DefaultPredictor(cfg)
    evaluator = COCOEvaluator("waymococo2_val", cfg, False, output_dir=outputpath)
    val_loader = build_detection_test_loader(cfg, "waymococo2_val")
    inference_on_dataset(predictor.model, val_loader, evaluator)

    # os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # trainer = Trainer(cfg)#DefaultTrainer(cfg) 
    # trainer.resume_or_load(resume=True)
    # trainer.train()