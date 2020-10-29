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



 # Implement a mapper, similar to the default DatasetMapper, but with customizations:
# def mapper(dataset_dict, cfg, is_train=True):
#     # Here we implement a minimal mapper for instance detection/segmentation
#     print("Using our own mapper!")
#     augmentation = build_augmentation(cfg, is_train)
    
#     dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
#     image = utils.read_image(dataset_dict["file_name"], format="BGR")
#     #image, transforms = T.apply_transform_gens(self.augmentation, image)
#     image_shape = image.shape[:2]  # h, w
    
#     dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1))
#     annos = [
#         utils.transform_instance_annotations(obj, transforms, image.shape[:2])
#         for obj in dataset_dict.pop("annotations")
#     ]
#     dataset_dict["instances"] = utils.annotations_to_instances(annos, image.shape[:2])
#     return dataset_dict


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
        #dataloader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True)) #https://github.com/facebookresearch/detectron2/blob/63f11718c68f1ae951caee157b4e10fae4d7e4be/projects/DensePose/densepose/data/dataset_mapper.py
        #T.Augmentation
        #dataloader = build_detection_train_loader(cfg, mapper=DatasetMapper(cfg, is_train=True, augmentations=[]))
        dataloader = build_detection_train_loader(cfg, mapper=mapper)
        #data_loader = build_detection_train_loader(cfg, mapper=mapper)
        #dataloader = build_detection_train_loader(cfg, mapper=MyDatasetMapper(cfg, is_train=True))
        #dataloader = build_detection_train_loader(cfg, mapper=NewDatasetMapper(cfg, is_train=True))
        #https://detectron2.readthedocs.io/_modules/detectron2/data/detection_utils.html#build_augmentation
        #https://github.com/facebookresearch/detectron2/blob/63f11718c68f1ae951caee157b4e10fae4d7e4be/detectron2/data/transforms/augmentation_impl.py
        return dataloader

def build_augmentation(cfg, is_train):
        logger = logging.getLogger(__name__)
#         if is_train:
#             min_size = cfg.INPUT.MIN_SIZE_TRAIN
#             max_size = cfg.INPUT.MAX_SIZE_TRAIN
#             sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
#         else:
#             min_size = cfg.INPUT.MIN_SIZE_TEST
#             max_size = cfg.INPUT.MAX_SIZE_TEST
#             sample_style = "choice"
#         augmentation = [T.ResizeShortestEdge(min_size, max_size, sample_style)]
        augmentation=[]
        if is_train:
            #augmentation.append(T.RandomFlip())
            augmentation.append(T.RandomBrightness(0.1,1.6))
#             augmentation.append(T.RandomContrast(0.1, 1))
#             augmentation.append(T.RandomSaturation(0.1, 1))
            #augmentation.append(T.RandomRotation(angle=[0, 90]))
            #augmentation.append(T.RandomCrop('relative_range', (0.4, 0.6)))
        
        result = augmentation #utils.build_augmentation(cfg, is_train)
#         if is_train:
#             random_rotation = T.RandomRotation(
#                 cfg.INPUT.ROTATION_ANGLES, expand=False, sample_style="choice"
#             )
#             result.append(random_rotation)
            #logger.info("DensePose-specific augmentation used in training: " + str(random_rotation))
        
        return result
    
#https://detectron2.readthedocs.io/tutorials/data_loading.html#write-a-custom-dataloader
class MyDatasetMapper:
    """
    A customized version of `detectron2.data.DatasetMapper`
    """
    
    def __init__(self, cfg, is_train=True):
        self.augmentation = build_augmentation(cfg, is_train)

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        
        assert not cfg.MODEL.LOAD_PROPOSALS, "not supported yet"

        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.
        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        image, transforms = T.apply_transform_gens(self.augmentation, image)
        image_shape = image.shape[:2]  # h, w
        dataset_dict["image"] = torch.as_tensor(image.transpose(2, 0, 1).astype("float32"))

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            return dataset_dict
        
        annos = [
            utils.transform_instance_annotations(obj, transforms, image.shape[:2])
            for obj in dataset_dict.pop("annotations")
        ]
        dataset_dict["instances"] = utils.annotations_to_instances(annos, image.shape[:2])

#         # USER: Implement additional transformations if you have other types of data
#         # USER: Don't call transpose_densepose if you don't need
#         annos = [
#             self._transform_densepose(
#                 utils.transform_instance_annotations(
#                     obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
#                 ),
#                 transforms,
#             )
#             for obj in dataset_dict.pop("annotations")
#             if obj.get("iscrowd", 0) == 0
#         ]

#         instances = utils.annotations_to_instances(annos, image_shape, mask_format="bitmask")

#         dataset_dict["instances"] = instances[instances.gt_boxes.nonempty()]
        return dataset_dict


class CutMix(Transform):
    
    def __init__(self, box_size=50, prob_cutmix=0.5):
        super().__init__()
        
        self.box_size = box_size
        self.prob_cutmix = prob_cutmix
        
    def apply_image(self, img):
        
        if random.random() > self.prob_cutmix:
            
            h, w = img.shape[:2]
            num_rand = np.random.randint(10, 20)
            for num_cut in range(num_rand):
                x_rand, y_rand = random.randint(0, w-self.box_size), random.randint(0, h-self.box_size)
                img[x_rand:x_rand+self.box_size, y_rand:y_rand+self.box_size, :] = 0
        
        return np.asarray(img)
    
    def apply_coords(self, coords):
        return coords.astype(np.float32)
    
class NewDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE)
            logging.getLogger(__name__).info("CropGen used in training: " + str(self.crop_gen))
        else:
            self.crop_gen = None

#         self.tfm_gens = utils.build_transform_gen(cfg, is_train)
        self.tfm_gens = [T.RandomBrightness(0.1, 1.6),
                         T.RandomContrast(0.1, 3),
                         T.RandomSaturation(0.1, 2),
                         T.RandomRotation(angle=[90, 90]),
                         T.RandomFlip(prob=0.4, horizontal=False, vertical=True),
                         T.RandomCrop('relative_range', (0.4, 0.6)),
                         CutMix()
                        ]

        # fmt: off
        self.img_format     = cfg.INPUT.FORMAT
        self.mask_on        = cfg.MODEL.MASK_ON
        self.mask_format    = cfg.INPUT.MASK_FORMAT
        self.keypoint_on    = cfg.MODEL.KEYPOINT_ON
        self.load_proposals = cfg.MODEL.LOAD_PROPOSALS
        # fmt: on
        if self.keypoint_on and is_train:
            # Flip only makes sense in training
            self.keypoint_hflip_indices = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)
        else:
            self.keypoint_hflip_indices = None

        if self.load_proposals:
            self.min_box_side_len = cfg.MODEL.PROPOSAL_GENERATOR.MIN_SIZE
            self.proposal_topk = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        self.is_train = is_train

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
        utils.check_image_size(dataset_dict, image)

        if "annotations" not in dataset_dict:
            image, transforms = T.apply_transform_gens(
                ([self.crop_gen] if self.crop_gen else []) + self.tfm_gens, image
            )
        else:
            # Crop around an instance if there are instances in the image.
            # USER: Remove if you don't use cropping
            if self.crop_gen:
                crop_tfm = utils.gen_crop_transform_with_instance(
                    self.crop_gen.get_crop_size(image.shape[:2]),
                    image.shape[:2],
                    np.random.choice(dataset_dict["annotations"]),
                )
                image = crop_tfm.apply_image(image)
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            if self.crop_gen:
                transforms = crop_tfm + transforms

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # USER: Remove if you don't use pre-computed proposals.
        if self.load_proposals:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, self.min_box_side_len, self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                if not self.mask_on:
                    anno.pop("segmentation", None)
                if not self.keypoint_on:
                    anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(
                    obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
                )
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(
                annos, image_shape, mask_format=self.mask_format
            )
            # Create a tight bounding box from masks, useful when image is cropped
            if self.crop_gen and instances.has("gt_masks"):
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = utils.filter_empty_instances(instances)

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            with PathManager.open(dataset_dict.pop("sem_seg_file_name"), "rb") as f:
                sem_seg_gt = Image.open(f)
                sem_seg_gt = np.asarray(sem_seg_gt, dtype="uint8")
            sem_seg_gt = transforms.apply_segmentation(sem_seg_gt)
            sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))
            dataset_dict["sem_seg"] = sem_seg_gt
        return dataset_dict

extrakeys=["truncation", "occlusion"]       
# train_annotation_json='/data/cmpe295-liu/Waymo/WaymoCOCOsmall/Training/annotations.json'
# train_images='/data/cmpe295-liu/Waymo/WaymoCOCOsmall/Training'
#BaseFolder='/data/cmpe295-liu/Waymo/WaymoCOCOtest/'
BaseFolder='/data/cmpe295-liu/Waymo/WaymoCOCO/'
train_annotation_json=os.path.join(BaseFolder, "Training/annotations.json")
train_images=os.path.join(BaseFolder, "Training")
# train_annotation_json='/data/cmpe295-liu/Waymo/WaymoCOCOtest/Training/annotations.json'
# train_images='/data/cmpe295-liu/Waymo/WaymoCOCOtest/Training'

# val_annotation_json='/data/cmpe295-liu/Waymo/WaymoCOCOsmall/Validation/annotations.json'
# val_images='/data/cmpe295-liu/Waymo/WaymoCOCOsmall/Validation'
train_annotation_json=os.path.join(BaseFolder, "Validation/annotations.json")
train_images=os.path.join(BaseFolder, "Validation")
#dataset_dicts = load_coco_json(train_annotation_json,train_images, "mywaymo_dataset_train", extrakeys)

for d in ["Training", "Validation"]:
    #DatasetCatalog.register("myuav1_" + d, lambda d=d: load_mycoco_json("/data/cmpe295-liu/UAVision/VisDrone2019-DET-" + d + "/annotations.json", "/data/cmpe295-liu/UAVision/VisDrone2019-DET-" + d + "/images", extrakeys))
    DatasetCatalog.register("mywaymo1_" + d, lambda d=d: load_coco_json(BaseFolder + d + "/annotations.json", BaseFolder + d + "/"))

FULL_LABEL_CLASSES = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']#['ignored-regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle','bus',  'motor', 'others']
for d in ["Training", "Validation"]:
    MetadataCatalog.get("mywaymo1_" + d).set(thing_classes=FULL_LABEL_CLASSES)
# from detectron2.data.datasets import register_coco_instances
# register_coco_instances("myuav1_train", {}, "/data/cmpe295-liu/UAVision/VisDrone2019-DET-train/annotations.json", "/data/cmpe295-liu/UAVision/VisDrone2019-DET-train/images")


cfg = get_cfg()
cfg.OUTPUT_DIR='./output_waymo' #'./output_x101'
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")) #faster_rcnn_R_101_FPN_3x.yaml, faster_rcnn_X_101_32x8d_FPN_3x
cfg.DATASETS.TRAIN = ("mywaymo1_Training",)
cfg.DATASETS.TEST = ("mywaymo1_Validation",)
cfg.DATALOADER.NUM_WORKERS = 1 #2
#cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
#cfg.MODEL.WEIGHTS = os.path.join('/home/010796032/PytorchWork', "fasterrcnn_x101_fpn_model_final_68b088.pkl")#using the local 
cfg.MODEL.WEIGHTS = os.path.join('/home/010796032/PytorchWork/output', "model_0079999.pth")
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

os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
trainer = Trainer(cfg)#DefaultTrainer(cfg) 
trainer.resume_or_load(resume=True)
trainer.train()

