import MyDetector.transforms as T
import utils
from PIL import Image
from glob import glob
import sys
import torch
import torch.utils.data as data

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN

#from MyDetector.engine import train_one_epoch, evaluate
import datetime
import os

import importlib
from MyDetector import WaymoDataset
importlib.reload(WaymoDataset)
import MyDetector.utils as utils

from MyDetector import engine
#importlib.reload(engine)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def load_previous_object_detection_model(num_classes, modelpath):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    model.load_state_dict(torch.load(modelpath))

    return model

class TorchVisionWaymoCOCOEvaluator(object):
    #args.showfig
    #args.modelname
    #args.modelbasefolder
    #args.modelfilename
        #args.FULL_LABEL_CLASSES
    #self.args.datasetpath="/data/cmpe295-liu/Waymo/WaymoCOCOsmall/"
    
    

    def __init__(self, args, waymovalidationframes):
        self.args = args
        use_cuda = True
        
        num_classes = len(args.FULL_LABEL_CLASSES) #Unknown:0, Vehicles: 1, Pedestrians: 2, Cyclists: 3, Signs (removed)
        modelpath=os.path.join(args.modelbasefolder, args.modelfilename)#'./saved_models_py4/model_18.pth'
        self.model=load_previous_object_detection_model(num_classes,modelpath) #./saved_models_mac1/model_27.pth
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        
        # move model to the right device
        self.model.to(self.device)
        self.model.eval()
        
#         print("Loading Waymo validation frames...")
#         waymovalidationframes=WaymoDataset.loadWaymoValidationFrames(self.args.datasetpath)
#         #mywaymovaldataset = myWaymoTestDataset(PATH, waymovalidationframes, get_transform(train=False))
        print("Total validation frames: ", len(waymovalidationframes))

        #mywaymovaldataset = WaymoDataset.myWaymoTestDataset(self.args.datasetpath, waymovalidationframes, get_transform(train=False))
        mywaymovaldataset = WaymoDataset.myNewWaymoDataset(self.args.datasetpath, waymovalidationframes, get_transform(train=False))
        self.valdata_loader = torch.utils.data.DataLoader(
            mywaymovaldataset, batch_size=1, shuffle=True, num_workers=4,
            collate_fn=utils.collate_fn)
        
        dataiter = iter(self.valdata_loader)
        images, labels = dataiter.next()
        print(labels)
        img, targets=self.valdata_loader.dataset[0]
        print("Iterate:", img.shape)
        #print(targets)
        bboxes = targets["boxes"]
        print(bboxes)
        bboxes[:, 2:] -= bboxes[:, :2]
        print(bboxes)
#         img, targets = ds[img_idx]
#         image_id = targets["image_id"].item()
#         img_dict = {}
#         img_dict['id'] = image_id
        #img_dict['height'] = img.shape[-2]
        #print(images.shape)
        

    def evaluate(self):
        #PATH='/data/cmpe295-liu/Waymo'
        
        engine.evaluate(self.model, self.valdata_loader, device=self.device)
        

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