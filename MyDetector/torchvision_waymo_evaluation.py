#import WaymoDataset
#from MyDetector import WaymoDataset
#from MyDetector import TorchVisionEvaluator
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

import transforms as T
import utils
from PIL import Image
from glob import glob
import sys
import torch
import torch.utils.data as data

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN

from engine import train_one_epoch, evaluate
import datetime
import os

class detectorargs:
    modelname = 'fasterrcnn_resnet50_fpn'#not used here
    modelbasefolder = '/home/010796032/Waymo/saved_models_py4/'
    modelfilename='model_23.pth'#'waymo_fasterrcnn_resnet50_fpnmodel_27.pth'
    showfig='False'
    FULL_LABEL_CLASSES = [
    'Unknown', 'Vehicles', 'Pedestrians', 'Cyclists'
    ]
    datasetpath='/data/cmpe295-liu/Waymo'#'/mnt/DATA5T/WaymoDataSet'
    #FULL_LABEL_CLASSES=['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']#['ignored-regions', 'pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle','bus',  'motor', 'others']

def loadWaymoValidationFrames(PATH, folderlist):
    validation_folders = folderlist #["validation_0000"]#validation_0004","validation_0003","validation_0002","validation_0001","validation_0000"]
    #validation_folders = ["validation_0000","validation_0001","validation_0002","validation_0003","validation_0004","validation_0005","validation_0006","validation_0007"] #["validation_0007","validation_0006","validation_0005","validation_0004","validation_0003","validation_0002","validation_0001","validation_0000"]
    data_files = [path for x in validation_folders for path in glob(os.path.join(PATH, x, "*.tfrecord"))]
    print(data_files)#all TFRecord file list
    print(len(data_files))
    dataset = [tf.data.TFRecordDataset(FILENAME, compression_type='') for FILENAME in data_files]#create a list of dataset for each TFRecord file
    frames = [] #store all frames = total number of TFrecord files * 40 frame(each TFrecord)
    for i, data_file in enumerate(dataset):
        print("Datafile: ",i)#Each TFrecord file
        for idx, data in enumerate(data_file): #Create frame based on Waymo API, 199 frames per TFrecord (20s, 10Hz)
#             if idx % 5 != 0: #Downsample every 5 images, reduce to 2Hz, total around 40 frames
#                 continue
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            frames.append(frame)
    return frames


def loadWaymoTestFrames(PATH):
    #train_folders = ["training_0005","training_0004","training_0003","training_0002","training_0001","training_0000"]#["training_0001"]# ["training_0000", "training_0001"]
    test_folders = ["testing_0007","testing_0006","testing_0005","testing_0004","testing_0003","testing_0002","testing_0001","testing_0000"]
    data_files = [path for x in test_folders for path in glob(os.path.join(PATH, x, "*.tfrecord"))]
    print(data_files)#all TFRecord file list
    print(len(data_files))
    dataset = [tf.data.TFRecordDataset(FILENAME, compression_type='') for FILENAME in data_files]#create a list of dataset for each TFRecord file
    frames = [] #store all frames = total number of TFrecord files * 40 frame(each TFrecord)
    for i, data_file in enumerate(dataset):
        print("Datafile: ",data_files[i])#Each TFrecord file
        for idx, data in enumerate(data_file): #Create frame based on Waymo API, 199 frames per TFrecord (20s, 10Hz)
#             if idx % 5 != 0: #Downsample every 5 images, reduce to 2Hz, total around 40 frames
#                 continue
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            frames.append(frame)
    return frames

class myNewWaymoDataset(data.Dataset): #Inheritance of data.Dataset, torch.utils.data
    def __init__(self, root, waymoframes, transforms=None):
        self.label_map = {0:0, 1: 1, 2:2, 4:3} #4 object types (remove sign)
        self.root = root
        self.transforms = transforms
        self.frames = waymoframes
        del_indexes = []
        #filter out frames with small or no annotations
        for i, frame in enumerate(self.frames): #Total number of TFrecord files * (Each TFrecord file, 40 frames)
            #print("lkk frame num: ",i)
            target = {}
            target_bbox = []
            target_labels = []
            target_areas = []
        
            for camera_labels in frame.camera_labels: #5 cameras
                #print(open_dataset.CameraName.Name.Name(camera_labels.name))
                if camera_labels.name != 1: #Only use front camera
                    continue
                    
                for label in camera_labels.labels:                    
                    xmin= label.box.center_x - 0.5 * label.box.length
                    ymin = label.box.center_y - 0.5 * label.box.width
                    xmax = xmin + label.box.length
                    ymax = ymin + label.box.width
                    area = label.box.length * label.box.width
                    
                    if xmin<=xmax and ymin<=ymax and xmin>=0 and ymin>=0: # and xmax<=1920 and ymax<=1280:# and area>2000:
                        target_bbox.append([xmin, ymin, xmax, ymax])
                        target_labels.append(self.label_map[label.type])
                        target_areas.append(area)
                        if (label.type==3):#traffic sign
                            print("label.type==3")
                        if (label.type==0):#unknow
                            print("label.type==1")
                    else:
                        print("drop frame: ", i)
                        print([xmin, ymin, xmax, ymax])#some xmax=1920.00002
                    
            target['boxes'] = torch.as_tensor(target_bbox, dtype=torch.float32)
            target['labels'] = torch.as_tensor(np.array(target_labels), dtype=torch.int64)
            target['image_id'] = torch.tensor([int(frame.context.name.split("_")[-2] + str(i))])
            target["area"] = torch.as_tensor(target_areas, dtype=torch.float32)
            target["iscrowd"] = torch.zeros((len(target['boxes'])), dtype=torch.int64)

            if len(target_bbox) > 0:
                #self.targets.append(target)
                testlabel=target_labels
                #print(target_labels)
                #print(len(target_bbox))
            else:
                print("no bbox, drop")
                del_indexes.append(i)
        
        for index in sorted(del_indexes, reverse=True):
            del self.frames[index] #delete frames without bounding box
        
#         for index in sorted(del_indexes, reverse=True):
#             del self.frames[index] #delete frames without bounding box

        
    def __getitem__(self, index):
        #self.frames: #Total number of TFrecord files * (Each TFrecord file, 40 frames)
        frameitem = self.frames[index]
        numimg=tf.image.decode_jpeg(frameitem.images[0].image).numpy()
        
#         data = asarray(numimg)
        #print("Type: ", type(numimg))
#         print(type(data))
#         # summarize shape
        #print("numimg shape:", numimg.shape)


        img = Image.fromarray(numimg).convert("RGB")
#         print(type(img))
#         # summarize image details
#         print(img.mode)
#         print(img.size)
#         print(img.shape)
        
        #print(numimg.shape)#[1280 1920 3]
        imgwidth=numimg.shape[1]#1920
        imgheight=numimg.shape[0]#1280
        
        target = {}
        target_bbox = []
        target_labels = []
        target_areas = []
        for camera_labels in frameitem.camera_labels: #5 cameras
            #print(open_dataset.CameraName.Name.Name(camera_labels.name))
            if camera_labels.name != 1: #Only use front camera
                continue

            for label in camera_labels.labels:                    
                xmin= label.box.center_x - 0.5 * label.box.length
                ymin = label.box.center_y - 0.5 * label.box.width
                xmax = xmin + label.box.length
                ymax = ymin + label.box.width
                area = label.box.length * label.box.width
                
                if xmin<=xmax and ymin<=ymax and xmin>=0 and ymin>=0: # and xmax<=imgwidth and ymax<=imgheight:# and area>2000 :
                    target_bbox.append([xmin, ymin, xmax, ymax])
                    target_labels.append(self.label_map[label.type])
                    target_areas.append(area)
#                 else:
#                     print("drop: ", index)
#                     print([label.box.center_x, label.box.center_y])
#                     print([label.box.length, label.box.width])
#                     print([xmin, ymin, xmax, ymax])
#                     print(area)
                    #print(self.frames[i].images.)

        target['boxes'] = torch.as_tensor(target_bbox, dtype=torch.float32)
        target['labels'] = torch.as_tensor(np.array(target_labels), dtype=torch.int64)
        target['image_id'] = torch.tensor([int(frameitem.context.name.split("_")[-2] + str(index))])
        target["area"] = torch.as_tensor(target_areas, dtype=torch.float32)
        target["iscrowd"] = torch.zeros((len(target['boxes'])), dtype=torch.int64)
                
        #target = self.targets[index]
        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target #image and annotations

    def __len__(self):
        return len(self.frames) #return size
    

class myWaymoTestDataset(data.Dataset): #Inheritance of data.Dataset, torch.utils.data
    def __init__(self, root, waymotestframes, transforms=None):
        #self.label_map = {0:0, 1: 1, 2:2, 4:3} #4 object types (remove sign)
        self.root = root
        self.transforms = transforms
        self.frames = waymotestframes
        #del_indexes = []
        
    def __getitem__(self, index):
        #self.frames: #Total number of TFrecord files * (Each TFrecord file, 40 frames)
        frameitem = self.frames[index]
        numimg=tf.image.decode_jpeg(frameitem.images[0].image).numpy()
        img = Image.fromarray(numimg).convert("RGB")
        #print(numimg.shape)#[1280 1920 3]
        imgwidth=numimg.shape[1]#1920
        imgheight=numimg.shape[0]#1280
        
        print(frameitem.context.name)#Refer to dataset.proto for the data format. The context contains shared information among all frames in the scene.
        print(frameitem.timestamp_micros)
        
        return img#, target #image and annotations

    def __len__(self):
        return len(self.frames) #return size

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
        mywaymovaldataset = myNewWaymoDataset(self.args.datasetpath, waymovalidationframes, get_transform(train=False))
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
        
        evaluate(self.model, self.valdata_loader, device=self.device)

#Downloading: "https://download.pytorch.org/models/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth" to /home/010796032/.cache/torch/hub/checkpoints/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth
if __name__ == "__main__":
    print("Loading Waymo validation frames...")
    folderlist = ["validation_0000"]
    waymovalidationframes=loadWaymoValidationFrames(detectorargs.datasetpath, folderlist)
    #mywaymovaldataset = myWaymoTestDataset(PATH, waymovalidationframes, get_transform(train=False))
    print("Total validation frames: ", len(waymovalidationframes))

    myevaluator=TorchVisionWaymoCOCOEvaluator(detectorargs, waymovalidationframes)
    print("Start evaluation....")
    myevaluator.evaluate()

    # #model.load_state_dict(torch.load("saved_models/model_6.pth"))
    # num_classes = 4 #Unknown:0, Vehicles: 1, Pedestrians: 2, Cyclists: 3, Signs (removed)
    # model_path="/home/010796032/Waymo/saved_models_py4/model_27.pth"
    # model=load_previous_object_detection_model(num_classes,model_path)#'./saved_models_py4/model_18.pth') #./saved_models_mac1/model_27.pth

    # PATH='/data/cmpe295-liu/Waymo'
    # print("Loading Waymo validation frames...")
    # waymovalidationframes=loadWaymoValidationFrames(PATH)
    # #mywaymovaldataset = myWaymoTestDataset(PATH, waymovalidationframes, get_transform(train=False))
    # print("Total validation frames: ", len(waymovalidationframes))

    # mywaymovaldataset = myNewWaymoDataset(PATH, waymovalidationframes, get_transform(train=False))
    # # define training and validation data loaders
    # valdata_loader = torch.utils.data.DataLoader(
    #     mywaymovaldataset, batch_size=4, shuffle=True, num_workers=4,
    #     collate_fn=utils.collate_fn)
    
    # # PATH='/data/cmpe295-liu/Waymo'
    # # waymotestframes=loadWaymoTestFrames(PATH)
    # # mywaymotestdataset=myWaymoTestDataset(PATH, waymotestframes, get_transform(train=False))
    # # #mywaymotestdataset = myNewWaymoDataset(PATH, waymotestframes, get_transform(train=False))
    # # print("Total testing frames: ", len(mywaymotestdataset))
    # # testdata_loader = torch.utils.data.DataLoader(
    # #     mywaymotestdataset, batch_size=1, shuffle=False, num_workers=4,
    # #     collate_fn=utils.collate_fn)
    
    # # train on the GPU or on the CPU, if a GPU is not available
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # # move model to the right device
    # model.to(device)
    # model.eval()
    # print("Start evaluation...")
    # evaluate(model, valdata_loader, device=device)
