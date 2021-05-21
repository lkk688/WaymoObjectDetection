import tensorflow as tf
import os
import tensorflow.compat.v1 as tf
import math
import numpy as np
import itertools

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.pyplot import figure
from matplotlib.patches import Rectangle

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2

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


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def loadWaymoFrames(PATH):
    #train_folders = ["training_0005","training_0004","training_0003","training_0002","training_0001","training_0000"]#["training_0001"]# ["training_0000", "training_0001"]
    #train_folders = ["training_0007","training_0006"]
    #train_folders = ["training_0010","training_0007","training_0006","training_0005","training_0004","training_0003","training_0002","training_0001","training_0000"]
    train_folders = ["training_0031","training_0030","training_0029","training_0028","training_0027","training_0026","training_0025", "training_0024", "training_0023","training_0022","training_0021","training_0020","training_0019","training_0018","training_0017","training_0016","training_0015","training_0014","training_0013","training_0012","training_0011","training_0010","training_0009","training_0008","training_0007","training_0006","training_0005","training_0004","training_0003","training_0002","training_0001","training_0000"]
    data_files = [path for x in train_folders for path in glob(os.path.join(PATH, x, "*.tfrecord"))]
    print(data_files)#all TFRecord file list
    print(len(data_files))
    dataset = [tf.data.TFRecordDataset(FILENAME, compression_type='') for FILENAME in data_files]#create a list of dataset for each TFRecord file
    frames = [] #store all frames = total number of TFrecord files * 40 frame(each TFrecord)
    for i, data_file in enumerate(dataset):
        print("Datafile: ",i)#Each TFrecord file
        for idx, data in enumerate(data_file): #Create frame based on Waymo API, 199 frames per TFrecord (20s, 10Hz)
            if idx % 5 != 0: #Downsample every 5 images, reduce to 2Hz, total around 40 frames
                continue
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            frames.append(frame)
    return frames

def loadWaymoValidationFrames(PATH):
    #validation_folders = ["validation_0007"]#,"validation_0005"]
    validation_folders = ["validation_0000","validation_0001","validation_0002","validation_0003","validation_0004","validation_0005","validation_0006","validation_0007"] #["validation_0007","validation_0006","validation_0005","validation_0004","validation_0003","validation_0002","validation_0001","validation_0000"]
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
                    if xmin<=xmax and ymin<=ymax and xmin>=0 and ymin>=0:# and xmax<=1920 and ymax<=1280:# and area>2000:
                        target_bbox.append([xmin, ymin, xmax, ymax])
                        target_labels.append(self.label_map[label.type])
                        target_areas.append(area)
                        if (label.type==3):#traffic sign
                            print("label.type==3")
                        if (label.type==0):#unknow
                            print("label.type==1")
                    else:
                        print("drop frame: ", i)
#                         print([label.box.center_x, label.box.center_y])
#                         print([label.box.length, label.box.width])
#                         print([xmin, ymin, xmax, ymax])
                        #print(self.frames[i].images.)
                    
#             target['boxes'] = torch.as_tensor(target_bbox, dtype=torch.float32)
#             target['labels'] = torch.as_tensor(np.array(target_labels), dtype=torch.int64)
#             target['image_id'] = torch.tensor([int(frame.context.name.split("_")[-2] + str(i))])
#             target["area"] = torch.as_tensor(target_areas, dtype=torch.float32)
#             target["iscrowd"] = torch.zeros((len(target['boxes'])), dtype=torch.int64)
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
        img = Image.fromarray(numimg).convert("RGB")
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
                
                if xmin<=xmax and ymin<=ymax and xmin>=0 and ymin>=0 and xmax<=imgwidth and ymax<=imgheight:# and area>2000 :
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

    
def get_object_detection_model(num_classes):
    # load an instance segmentation model pre-trained on COCO
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
#     pretrained_backbone = False
#     trainable_backbone_layers=3
#     backbone = torchvision.models.resnet.resnet50('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)
#     model = FasterRCNN('resnet50', num_classes)
    pretrained= True
    if pretrained:
        model.load_state_dict(torch.load('./PytorchRCNN/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'))
        #model.load_state_dict(torch.load('./saved_models2/model_9.pth'))
        #state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'])
        #model.load_state_dict(state_dict)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

def get_previous_object_detection_model(num_classes, modelpath):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    pretrained= True
    if pretrained:
        #model.load_state_dict(torch.load('./PytorchRCNN/fasterrcnn_resnet50_fpn_coco-258fb6c6.pth'))
        model.load_state_dict(torch.load(modelpath))#'./saved_models2/model_9.pth'))
        #state_dict = load_state_dict_from_url(model_urls['fasterrcnn_resnet50_fpn_coco'])
        #model.load_state_dict(state_dict)

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

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

##Unknown:0, Vehicles: 1, Pedestrians: 2, Cyclists: 3, Signs (removed)
INSTANCE_CATEGORY_NAMES = [
    'Unknown', 'Vehicles', 'Pedestrians', 'Cyclists'
]
def get_prediction(modeluse, image, device, threshold):
    target = {}
    target_bbox = []
    target_labels = []
    target_areas = []
    img, target = get_transform(train=False)(image, target)
    pred = modeluse([img.to(device)])
    #pred = modeluse.forward([img.to(device)], [target])
    #print(pred)
    #pred=pred.cpu().numpy()
    
    #img = Image.open(img_path) # Load the image
#     transform = T.Compose([T.ToTensor()]) # Defing PyTorch Transform
#     img = transform(image) # Apply the transform to the image
#     pred = modeluse([img]) # Pass the image to the model
    
    pred_class = [INSTANCE_CATEGORY_NAMES[i] for i in list(pred[0]['labels'].cpu().numpy())] # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred[0]['boxes'].detach().cpu().numpy())] # Bounding boxes
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


INSTANCE_pb2 = {
    'Unknown':label_pb2.Label.TYPE_UNKNOWN, 'Vehicles':label_pb2.Label.TYPE_VEHICLE, 'Pedestrians':label_pb2.Label.TYPE_PEDESTRIAN, 'Cyclists':label_pb2.Label.TYPE_CYCLIST
}
#label_pb2.Label.TYPE_PEDESTRIAN

label_map = {0:0, 1: 1, 2:2, 4:3}
score_threshold = 0.5

def create_pd(frame, objmodel, device, score_threshold):
    """Creates a prediction objects file."""
    objects = metrics_pb2.Objects()
    
    image = tf.image.decode_jpeg(frame.images[0].image).numpy()#front camera image
    img = Image.fromarray(image)

    #print(frame.camera_labels)#no labels
    #print(frame.context.name)#Refer to dataset.proto for the data format. The context contains shared information among all frames in the scene.
    #print(frame.timestamp_micros)

    #run the prediction
    boxes, pred_cls, scores = get_prediction(objmodel, img, device, score_threshold)
#     print(pred_cls)
#     print(boxes)
    boxnum=min(len(boxes),400)
    for i in range(boxnum):#patch in pred_bbox:
        patch=boxes[i]
        label=pred_cls[i]
        #print(patch)#[(827.3006, 617.69965), (917.02795, 656.8029)]
        
        o = metrics_pb2.Object() #One frame: https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/protos/metrics.proto
        # The following 3 fields are used to uniquely identify a frame a prediction
        # is predicted at. Make sure you set them to values exactly the same as what
        # we provided in the raw data. Otherwise your prediction is considered as a
        # false negative.
        o.context_name = frame.context.name #('context_name for the prediction. See Frame::context::name ''in  dataset.proto.')
        # The frame timestamp for the prediction. See Frame::timestamp_micros in
        # dataset.proto.
        invalid_ts = frame.timestamp_micros #-1
        o.frame_timestamp_micros = int(invalid_ts)
        # This is only needed for 2D detection or tracking tasks.
        # Set it to the camera name the prediction is for.
        o.camera_name = dataset_pb2.CameraName.FRONT
        
        # Populating box and score.
        box = label_pb2.Label.Box() #Bounding box: https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/label.proto
        width=patch[1][0]-patch[0][0]
        height=patch[1][1]-patch[0][1]
        box.center_x = patch[0][0]+width/2
        box.center_y = patch[0][1]+height/2
        box.center_z = 0
        box.length = 0
        box.width = width
        box.height = height
        box.heading = 0
        
        o.object.box.CopyFrom(box) #o.object: Label type
        # This must be within [0.0, 1.0]. It is better to filter those boxes with
        # small scores to speed up metrics computation.
        o.score = 0.5
        # For tracking, this must be set and it must be unique for each tracked
        # sequence.
        o.object.id = 'xxx'#'unique object tracking ID'
        # Use correct type.
        o.object.type = INSTANCE_pb2[label]#label_pb2.Label.TYPE_PEDESTRIAN
        #print(o)
        objects.objects.append(o)
    
    return objects

    # Add more objects. Note that a reasonable detector should limit its maximum
    # number of boxes predicted per frame. A reasonable value is around 400. A
    # huge number of boxes can slow down metrics computation.
    
def generatevalidation(PATH, outputfilepath, MODEL_DIR):
    now = datetime.datetime.now()
    print ("In Training, current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))
    
    tf.enable_eager_execution()
    print(tf.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name())
    device = torch.device("cuda")
    
    print("Loading Waymo validation frames...")
    waymovalidationframes=loadWaymoValidationFrames(PATH)
    #mywaymovaldataset = myNewWaymoDataset(PATH, waymovalidationframes, get_transform(train=False))
    print("Total validation frames: ", len(waymovalidationframes))
    
    num_classes = 4 #Unknown:0, Vehicles: 1, Pedestrians: 2, Cyclists: 3, Signs (removed)
    # get the model using our helper function
    print("Loading previous model: " + MODEL_DIR)
    #model = get_previous_object_detection_model(num_classes, previous_model_path)
    model = load_previous_object_detection_model(num_classes, MODEL_DIR)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # move model to the right device
    model.to(device)
    model.eval()
    
    f = open(outputfilepath, 'wb') #waymovalidationframes, waymotestframes
    outputallframes=waymovalidationframes
    print("Total frames: ", len(outputallframes))
    for i in range(len(outputallframes)): #len(outputallframes)
        if i%10 ==0:
            print("current frame: ", i)
        frame = outputallframes[i]
        objects = create_pd(frame, model, device, score_threshold=0.5)
        #print(objects)
        # Write objects to a file.
        f.write(objects.SerializeToString())
    f.close()
    
    now = datetime.datetime.now()
    print ("Finished validation, current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))
    
    # print("Start testing")
    # waymotestframes=loadWaymoTestFrames(PATH)
    # print("Total testing frames: ", len(waymotestframes))
    
    # outputfilepathtest='saved_models_py4/py_saved_models_py4_full14_test1.bin'
    # f = open(outputfilepathtest, 'wb') #waymovalidationframes, waymotestframes
    # outputallframes=waymovalidationframes
    # print("Total frames: ", len(outputallframes))
    # for i in range(len(outputallframes)): #len(outputallframes)
    #     if i%20 ==0:
    #         print("current frame: ", i)
    #     frame = outputallframes[i]
    #     objects = create_pd(frame, model, device, score_threshold=0.5)
    #     #print(objects)
    #     # Write objects to a file.
    #     f.write(objects.SerializeToString())
    # f.close()

from waymo_open_dataset.protos import metrics_pb2,submission_pb2
from waymo_open_dataset import dataset_pb2 
from waymo_open_dataset import label_pb2    
def generatevalidationsubmission(PATH, outputfilepath, MODEL_DIR):
    now = datetime.datetime.now()
    print ("In generatevalidationsubmission, current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))
    
    tf.enable_eager_execution()
    print(tf.__version__)
    print(torch.cuda.is_available())
    print(torch.cuda.device_count())
    print(torch.cuda.get_device_name())
    device = torch.device("cuda")
    
    print("Loading Waymo validation frames...")
    waymovalidationframes=loadWaymoValidationFrames(PATH)
    #mywaymovaldataset = myNewWaymoDataset(PATH, waymovalidationframes, get_transform(train=False))
    print("Total validation frames: ", len(waymovalidationframes))
    
    num_classes = 4 #Unknown:0, Vehicles: 1, Pedestrians: 2, Cyclists: 3, Signs (removed)
    # get the model using our helper function
    print("Loading previous model: " + MODEL_DIR)
    #model = get_previous_object_detection_model(num_classes, previous_model_path)
    model = load_previous_object_detection_model(num_classes, MODEL_DIR)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # move model to the right device
    model.to(device)
    model.eval()
    
    objects = metrics_pb2.Objects()
    f = open(outputfilepath, 'wb') #waymovalidationframes, waymotestframes
    outputallframes=waymovalidationframes
    print("Total frames: ", len(outputallframes))
    #step=5
    for i in range(len(outputallframes)): #len(outputallframes)
        if i%10 ==0:
            print("current frame: ", i)
        frame = outputallframes[i]
        image = tf.image.decode_jpeg(frame.images[0].image).numpy()#front camera image
        img = Image.fromarray(image)
        boxes, pred_cls, scores = get_prediction(model, img, device, score_threshold)
        total_boxes=len(boxes)
        if len(boxes) == 0:
            continue
        for i in range(total_boxes):#patch in pred_bbox:
            label=pred_cls[i]
            bbox=boxes[i]
            score = scores[i]
            o = metrics_pb2.Object()
            o.context_name = frame.context.name
            o.frame_timestamp_micros = int(frame.timestamp_micros)
            o.camera_name = dataset_pb2.CameraName.FRONT
            o.score = score
            
            # Populating box and score.
            box = label_pb2.Label.Box()
            box.length = bbox[1][0] - bbox[0][0]
            box.width = bbox[1][1] - bbox[0][1]
            box.center_x = bbox[0][0] + box.length * 0.5
            box.center_y = bbox[0][1] + box.width * 0.5

            o.object.box.CopyFrom(box)
            o.object.detection_difficulty_level = label_pb2.Label.LEVEL_1
            o.object.num_lidar_points_in_box = 100
            o.object.type = INSTANCE_pb2[label]# INSTANCE_CATEGORY_NAMES.index(label) #INSTANCE_pb2[label]
            print(f'Object type label: {label}, {INSTANCE_pb2[label]}, {INSTANCE_CATEGORY_NAMES.index(label)}')
            assert o.object.type != label_pb2.Label.TYPE_UNKNOWN
            objects.objects.append(o)
    
    submission = submission_pb2.Submission()
    submission.task = submission_pb2.Submission.DETECTION_2D 
    submission.account_name = 'kaikai.liu@sjsu.edu'
    submission.authors.append('Kaikai Liu')
    submission.affiliation = 'None'
    submission.unique_method_name = 'torchvisionfaster'
    submission.description = 'none'
    submission.method_link = "empty method"
    submission.sensor_type = submission_pb2.Submission.CAMERA_ALL
    submission.number_past_frames_exclude_current = 0
    submission.number_future_frames_exclude_current = 0
    submission.inference_results.CopyFrom(objects)
    f = open(outputfilepath, 'wb')
    #f = open("./drive/My Drive/waymo_submission/waymo35.bin", 'wb')
    f.write(submission.SerializeToString())
    f.close()
    
    now = datetime.datetime.now()
    print ("Finished validation, current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == "__main__":
    PATH='/data/cmpe295-liu/Waymo'
    EPOCHS=40
    previous_model_path = '/home/010796032/Waymo/saved_models_py4/model_27.pth'
    #previous_model_path = './saved_models_py4/model_14.pth'
    #outputfilepath='/home/010796032/MyRepo/submissionoutput/py_saved_models_py4_full14_val2.bin' #'/tmp/your_preds.bin'
    outputfilepath='/home/010796032/MyRepo/submissionoutput/torchvision_model27_valnewfull.bin' #'/tmp/your_preds.bin'
   
    #generatevalidation(PATH, outputfilepath, previous_model_path)
    generatevalidationsubmission(PATH, outputfilepath, previous_model_path)