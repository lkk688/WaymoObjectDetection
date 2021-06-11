import torch
import torch.utils.data as data

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN

import datetime
import os
import numpy as np

# Global variables that hold the models
model = None
DATA_FIELDS = ['FRONT_IMAGE']
FILTERthreshold=0.2
classes=('vehicle', 'pedestrian', 'sign', 'cyclist')
num_classes=5 # ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']

#Downloading: "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth" to /home/lkk/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
def load_previous_object_detection_model(num_classes, modelpath):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    pretrained= True
    if pretrained:
        model.load_state_dict(torch.load(modelpath))#'./saved_models2/model_9.pth'))

    return model

modelname = 'torchvisionfasterrcnn'
config=""
model_dir = '/home/010796032/MyRepo/Torchoutput/fasterrcnntrain/model_6.pth'
device = None

from os import path
def setupmodeldir(model_path, config_path=''):
    global model_dir
    if path.exists(model_path):
        model_dir=model_path
        print(f'Setup new model path:{model_path}')
    if path.exists(config_path):
        global config
        config=config_path
        print(f'Setup new config path:{config}')

def initialize_model():
    """Initialize the global model variable to the pretrained EfficientDet.
    This assumes that the EfficientDet model has already been downloaded to a
    specific path, as done in the Dockerfile for this example.
    """
    
    global model
    global config
    global model_dir

    model = load_previous_object_detection_model(num_classes, model_dir)
    global device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # move model to the right device
    model.to(device)
    model.eval()

import random
from torchvision.transforms import functional as F
class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensor(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

def get_transform(train):
    transforms = []
    transforms.append(ToTensor())
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
    return Compose(transforms) 

from waymo_open_dataset import label_pb2
INSTANCE_pb2 = {
    1: label_pb2.Label.TYPE_VEHICLE, 4: label_pb2.Label.TYPE_CYCLIST, 2: label_pb2.Label.TYPE_PEDESTRIAN, 3: label_pb2.Label.TYPE_SIGN
}
# enum Type {
#     TYPE_UNKNOWN = 0;
#     TYPE_VEHICLE = 1;
#     TYPE_PEDESTRIAN = 2;
#     TYPE_SIGN = 3;
#     TYPE_CYCLIST = 4;
#   }
# INSTANCE_CATEGORY_NAMES = [
#     'Unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist'
# ]
#classes=('vehicle', 'pedestrian', 'sign', 'cyclist')
def get_prediction(modeluse, image, device, threshold):
    target = {}
    target_bbox = []
    target_labels = []
    target_areas = []
    img, target = get_transform(train=False)(image, target)
    pred = modeluse([img.to(device)])

    pred_class = [INSTANCE_pb2[i] for i in list(pred[0]['labels'].cpu().numpy())] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())] # Bounding boxes, xmin ymin xmax ymax 
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    #print("Length: %d, %d, %d", len(pred_class), len(pred_boxes), len(pred_score))
    predlist=[pred_score.index(x) for x in pred_score if x > threshold] # Get list of index with score greater than threshold.
    #print(predlist)
    #print(pred_score)
    #print(pred_class)
    if len(pred_score) == 0:
        return pred_boxes, pred_class, pred_score

    if len(predlist)>1:#1:
        pred_t = predlist[-1] 
        pred_boxes = pred_boxes[:pred_t+1]
        pred_class = pred_class[:pred_t+1]
        pred_score = pred_score[:pred_t+1]
    else:
        #print(pred_score)
        #print(pred_class)
        #if (len(pred_boxes)>0):
        maxpos = pred_score.index(max(pred_score))
        #print("max position:", maxpos)
        pred_boxes = pred_boxes[maxpos:maxpos+1]#get the first one
        pred_class = pred_class[maxpos:maxpos+1]#get the first one
        pred_score = pred_score[maxpos:maxpos+1]  
#             print(pred_boxes)
#             print(pred_class)
#         pred_boxes = pred_boxes[0]#get the first one
#         pred_class = pred_class[0]#get the first one
    
    #pred_boxes = [x.data.cpu().numpy() for idx, x in enumerate(pred[0]['boxes']) if pred[0]["scores"][idx] > score_threshold]
    #pred_class = [x.data.cpu().numpy() for idx, x in enumerate(pred[0]['labels']) if pred[0]["scores"][idx] > score_threshold]
    
    return pred_boxes, pred_class, pred_score

def run_model(**kwargs):
    """Run the model on the RGB image.
    Args:
      FRONT_IMAGE: H x W x 3 numpy ndarray.
    Returns:
      Dict from string to numpy ndarray.
    """
    # Run the model.
    frontimagekey=DATA_FIELDS[0]
    FRONT_IMAGE=kwargs[frontimagekey]
    #print(FRONT_IMAGE.size)
    imageshape=FRONT_IMAGE.shape
    #print("imageshape:", imageshape)
    im_width=imageshape[1]#1920
    im_height=imageshape[0]#1280

    boxes, pred_cls, scores = get_prediction(model, FRONT_IMAGE, device, FILTERthreshold)
    numbox=len(pred_cls)
    if numbox>0:
        newboxes=[]
        #[[xmin ymin] [xmax ymax]]  to (center_x, center_y, width, height) in image size
        #pred_boxes = [[(i[1]*im_width, i[0]*im_height), (i[3]*im_width, i[2]*im_height)] for i in list(pred_boxes)] # Bounding boxes
        for index_i in range(numbox):
            currentbox=boxes[index_i]
            xmin=currentbox[0][0]
            ymin=currentbox[0][1]
            xmax=currentbox[1][0]
            ymax=currentbox[1][1]
            center_x = (xmin+xmax)/2
            center_y = (ymin+ymax)/2
            length = xmax-xmin
            width = ymax-ymin
            newboxes.append([center_x, center_y, length, width])

        return {
            'boxes':  np.array(newboxes),
            'scores': np.array(scores),
            'classes': np.array(pred_cls).astype(np.uint8),
        }
    else:#empty
        return {
            'boxes': np.array(boxes),
            'scores': np.array(scores),
            'classes': np.array(pred_cls).astype(np.uint8),
        }
    # return {
    #         'boxes': np.array(boxes),
    #         'scores': np.array(scores),
    #         'classes': np.array(pred_cls).astype(np.uint8),
    #     }