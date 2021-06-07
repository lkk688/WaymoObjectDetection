import transforms as T
from pycocotools.coco import COCO
import torchvision
import torch
import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
from PIL import Image
from glob import glob

import math
import itertools

import torch
import torch.utils.data as data
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())
#device = torch.device("cuda")

import utils
# def get_transform(train):
#     transforms = []
#     transforms.append(T.ToTensor())
#     if train:
#         transforms.append(T.RandomHorizontalFlip(0.5))
#     return T.Compose(transforms)

def get_transform():
    custom_transforms = []
    custom_transforms.append(torchvision.transforms.ToTensor())
    return torchvision.transforms.Compose(custom_transforms)


class myCOCODataset(torch.utils.data.Dataset):
    def __init__(self, root, annotation, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotation)
        self.ids = list(sorted(self.coco.imgs.keys()))#

        #
        dataset=self.coco.dataset
        imgToAnns=self.coco.imgToAnns
        catToImgs =self.coco.catToImgs
        cats=self.coco.cats

    
    def _get_target(self, id):
        'Get annotations for sample'

        # List: get annotation id from coco
        ann_ids = self.coco.getAnnIds(imgIds=id)
        # Dictionary: target coco_annotation file for an image
        #ref: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
        annotations = self.coco.loadAnns(ann_ids)

        boxes, categories = [], []
        for ann in annotations:
            if ann['bbox'][2] < 1 and ann['bbox'][3] < 1:
                continue
            boxes.append(ann['bbox'])
            cat = ann['category_id']
            if 'categories' in self.coco.dataset:
                cat = self.categories_inv[cat]
            categories.append(cat)

        if boxes:
            target = (torch.FloatTensor(boxes),
                      torch.FloatTensor(categories).unsqueeze(1))
        else:
            target = (torch.ones([1, 4]), torch.ones([1, 1]) * -1)

        return target


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: Tuple (image, target). target is the object returned by ``coco.loadAnns``.
        """
        # Own coco file
        coco = self.coco
        # Image ID
        img_id = self.ids[index]
        imginfo=self.coco.imgs[img_id]
        path = imginfo['file_name']
        #print(f'index: {index}, img_id:{img_id}, info: {imginfo}')

        # path for input image
        #loadedimglist=coco.loadImgs(img_id)
        # print(loadedimglist)
        #path = coco.loadImgs(img_id)[0]['file_name']
        #print("image path:", path)
        # open the input image
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        #img = Image.open(os.path.join(self.root, path)).convert('RGB')


        # List: get annotation id from coco
        #ann_ids = coco.getAnnIds(imgIds=img_id)
        annolist=[self.coco.imgToAnns[img_id]]
        anns = list(itertools.chain.from_iterable(annolist))
        ann_ids = [ann['id'] for ann in anns]
        # Dictionary: target coco_annotation file for an image
        #ref: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py
        targets  = coco.loadAnns(ann_ids)
        #targets=self.anns[ann_ids]
        #print("targets:", targets)
        
        #image_id = targets["image_id"].item()

        # number of objects in the image
        num_objs = len(targets)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        target = {}
        target_bbox = []
        target_labels = []
        target_areas = []
        target_crowds = []
        for i in range(num_objs):
            xmin = targets[i]['bbox'][0]
            ymin = targets[i]['bbox'][1]
            width=targets[i]['bbox'][2]
            xmax = xmin + width
            height = targets[i]['bbox'][3]
            ymax = ymin + height
            if xmin<=xmax and ymin<=ymax and xmin>=0 and ymin>=0 and width>1 and height>1:
                target_bbox.append([xmin, ymin, xmax, ymax])
                target_labels.append(targets[i]['category_id'])
                target_crowds.append(targets[i]['iscrowd'])
                target_areas.append(targets[i]['area'])
        num_objs=len(target_bbox)
        #print("target_bbox len:", num_objs)
        if num_objs>0:
            #print("target_labels:", target_labels)
            target['boxes'] = torch.as_tensor(target_bbox, dtype=torch.float32)
            # Labels int value for class
            target['labels'] = torch.as_tensor(np.array(target_labels), dtype=torch.int64)
            target['image_id'] = torch.tensor([int(img_id)])
            #torch.tensor([int(frameitem.context.name.split("_")[-2] + str(index))])
            target["area"] = torch.as_tensor(np.array(target_areas), dtype=torch.float32)
            target["iscrowd"] = torch.as_tensor(np.array(target_crowds), dtype=torch.int64)#torch.zeros((len(target['boxes'])), dtype=torch.int64)
        else:
            #negative example, ref: https://github.com/pytorch/vision/issues/2144
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)#not empty
            target['labels'] = torch.as_tensor(np.array(target_labels), dtype=torch.int64)#empty
            target['image_id'] = torch.tensor([int(img_id)])
            target["area"] = torch.as_tensor(np.array(target_areas), dtype=torch.float32)#empty
            target["iscrowd"] = torch.as_tensor(np.array(target_crowds), dtype=torch.int64)#empty

        if self.transforms is not None:
            img = self.transforms(img)
        #print("target:", target)
        return img, target

    def __len__(self):
        return len(self.ids)

import matplotlib.pyplot as plt
import matplotlib.patches as patches
#%matplotlib inline

def myshowimage_torchdataset(dataset, layout, cmap=None):
    """Show a camera image and the given camera labels."""

    ax = plt.subplot(*layout)

    imgdata=dataset[0].permute(1, 2, 0)
    imgdata.shape
    
    boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(dataset[1]['boxes'].detach().numpy())]
    for i in range(len(boxes)): #[xmin, ymin, xmax, ymax], 1280, 1920
        ax.add_patch(patches.Rectangle(
            xy=boxes[i][0], #(boxes[i][0][0], boxes[i][0][1]),
            width=boxes[i][1][0]-boxes[i][0][0],#x-axis
            height=boxes[i][1][1]-boxes[i][0][1],#y-axis
            linewidth=1,
            edgecolor='red',
            facecolor='none'))                                                                                                                       

    # Show the camera image.
    plt.imshow(imgdata, cmap=cmap)
    #plt.title(open_dataset.CameraName.Name.Name(camera_image.name))
    plt.grid(True)
    #plt.axis('off')
    plt.savefig("annotationtorch.png")

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.faster_rcnn import FasterRCNN
#from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor

def load_previous_object_detection_model(num_classes, modelpath):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    model.load_state_dict(torch.load(modelpath))

    return model

def get_object_detection_model(num_classes, modelpath, pretrained= True):
    # load an instance segmentation model pre-trained on COCO
    #model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    
#     pretrained_backbone = False
#     trainable_backbone_layers=3
#     backbone = torchvision.models.resnet.resnet50('resnet50', pretrained_backbone, trainable_layers=trainable_backbone_layers)
#     model = FasterRCNN('resnet50', num_classes)
    #pretrained= True
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

def load_previous_object_detection_model_new(num_classes, modelpath, new_num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    pretrained= True
    if pretrained:
        model.load_state_dict(torch.load(modelpath))#'./saved_models2/model_9.pth'))

    # get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, new_num_classes)

    return model

if __name__ == '__main__':
    # path to your own data and coco file
    data_root = '/data/cmpe249-f20/WaymoCOCOMulti/trainvalall/'

    #data_root = '/DATA5T/Dataset/WaymoCOCO/'
    ann_file = data_root + 'annotations_train684step8allobject.json'#'annotations_train20new.json'
    # create own Dataset
    mywaymodataset = myCOCODataset(root=data_root,  
                          annotation=ann_file,
                          transforms=get_transform()
                          )
    print("Dataset",len(mywaymodataset))#199935

    # split the dataset in train and test set
    indices = torch.randperm(len(mywaymodataset)).tolist()
    idxsplit=int(len(indices)*0.80)#159948
    dataset_train = torch.utils.data.Subset(mywaymodataset, indices[:idxsplit])
    mywaymodataset.transforms = get_transform()
    dataset_test = torch.utils.data.Subset(mywaymodataset, indices[idxsplit+1:])
    #print(indices[idxsplit+1:])
    # dataset_train = torch.utils.data.Subset(dataset, indices[:-100])
    # dataset.transforms = get_transform(train=False)
    # dataset_test = torch.utils.data.Subset(dataset, indices[-100:])
    print (len(mywaymodataset))
    print (len(dataset_test))#39986


    #import vision.references.detection.utils as utils
    BATCH_SIZE=8
    # define training and validation data loaders
    data_loader = torch.utils.data.DataLoader(
        dataset_train, batch_size=BATCH_SIZE, shuffle=True, num_workers=0,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1, shuffle=False, num_workers=0,
        collate_fn=utils.collate_fn)

    # DataLoader is iterable over Dataset
    # imgs, annotations=next(iter(data_loader_test))
    # print(annotations)

    myshowimage_torchdataset(mywaymodataset[0],[1,1,1])
    # for imgs, annotations in testdata: #data_loader.take(2):
    #     imgs = list(img.to(device) for img in imgs)
    #     annotations = [{k: v.to(device) for k, v in t.items()} for t in annotations]
    #     print(annotations)

    num_classes=5 # ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']
    previous_num_classes = 4 #Unknown:0, Vehicles: 1, Pedestrians: 2, Cyclists: 3, Signs (removed)
    #previous_model_path = '/Developer/MyRepo/mymodels/torchfasterrcnn/model_27.pth'
    previous_model_path = '/home/010796032/Waymo/saved_models_py4/model_27.pth'
    #print("Loading previous model: " + previous_model_path)
    #model = get_object_detection_model(num_classes, previous_model_path)#(num_classes, previous_model_path)
    #model = load_previous_object_detection_model_new(previous_num_classes, previous_model_path, num_classes)
    #continue training based on the same model
    previous_model_path = '/home/010796032/MyRepo/Torchoutput/fasterrcnntrain/model_0.pth'
    model = load_previous_object_detection_model(num_classes, previous_model_path)

    # select device (whether GPU or CPU)
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # move model to the right device
    model.to(device)

    # construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    # and a learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=3,
                                                gamma=0.1)
    #from vision.references.detection.engine import train_one_epoch, evaluate
    from engine import train_one_epoch, evaluate
    #evaluate(model, data_loader_test, device=device)

    import sys
    num_epochs=10
    MODELWORK_DIR = "/home/010796032/MyRepo/Torchoutput/fasterrcnntrain"
    CHECK_FOLDER = os.path.isdir(MODELWORK_DIR)
    # If folder doesn't exist, then create it.
    if not CHECK_FOLDER:
        os.makedirs(MODELWORK_DIR)
        print("created folder : ", MODELWORK_DIR)
    for epoch in range(num_epochs):
            # train for one epoch, printing every 10 iterations
            train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=5)#10
            # update the learning rate
            lr_scheduler.step()
            torch.save(model.state_dict(), os.path.join(MODELWORK_DIR, "model_%s.pth"%(epoch)))
            # evaluate on the test dataset
            evaluate(model, data_loader_test, device=device)