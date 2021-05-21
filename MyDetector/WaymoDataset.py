import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools
import utils
from PIL import Image
from glob import glob
import sys
import torch
import torch.utils.data as data
import datetime
import os

from numpy import asarray

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

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