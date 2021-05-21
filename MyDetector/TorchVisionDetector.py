import MyDetector.transforms as T
import utils
from PIL import Image
from glob import glob
import numpy as np
import sys
import torch
import torch.utils.data as data

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN

from MyDetector.engine import train_one_epoch, evaluate
import datetime
import os
from MyDetector.Postprocess import postfilter

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)

def load_previous_object_detection_model(num_classes, modelpath):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    model.load_state_dict(torch.load(modelpath))

    return model

class TorchVisionFasterRCNNDetector(object):
    #args.showfig
    #args.modelname
    #args.modelbasefolder
    #args.modelfilename
        #args.FULL_LABEL_CLASSES
    #self.args.datasetpath="/data/cmpe295-liu/Waymo/WaymoCOCOsmall/"
    
    

    def __init__(self, args):
        self.args = args
        use_cuda = True
        self.threshold = args.threshold if args.threshold is not None else 0.1

        self.FULL_LABEL_CLASSES=args.FULL_LABEL_CLASSES
        
        num_classes = len(args.FULL_LABEL_CLASSES) #Unknown:0, Vehicles: 1, Pedestrians: 2, Cyclists: 3, Signs (removed)
        modelpath=os.path.join(args.modelbasefolder, args.modelfilename)#'./saved_models_py4/model_18.pth'
        self.model=load_previous_object_detection_model(num_classes,modelpath) #./saved_models_mac1/model_27.pth
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # move model to the right device
        self.model.to(self.device)
        self.model.eval()

    def detect(self, image):
        target = {}
        target_bbox = []
        target_labels = []
        target_areas = []
        img, target = get_transform(train=False)(image, target)
        pred = self.model([img.to(self.device)])
    
        #pred_class = [INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())] # Get the Prediction Score
        pred_class = list(pred[0]['labels'].cpu().numpy()) #[i for i in list(pred[0]['labels'].cpu().numpy())] # Get the Prediction Score
        pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())] # Bounding boxes
        pred_score = list(pred[0]['scores'].detach().cpu().numpy())

        #Post filter based on threshold
        pred_boxes, pred_class, pred_score = postfilter(pred_boxes, pred_class, pred_score, self.threshold)
        
        return pred_boxes, pred_class, pred_score


#         bbox_xcycwh, cls_conf, cls_ids = [], [], []

#         #box format is XYXY_ABS
#         for (box, _class, score) in zip(boxes, classes, scores):
#             #if _class == 0: # the orignal code only track people?
#             x0, y0, x1, y1 = box
#             bbox_xcycwh.append([(x1 + x0) / 2, (y1 + y0) / 2, (x1 - x0), (y1 - y0)]) # convert to x-center, y-center, width, height
#             cls_conf.append(score)
#             cls_ids.append(_class)

#         return np.array(bbox_xcycwh, dtype=np.float64), np.array(cls_conf), np.array(cls_ids)

#     def detectwithvisualization(self, im)
#         pred_boxes, classes, scores=self.detect(im)