from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
# import some common detectron2 utilities
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
from detectron2.structures import BitMasks, Boxes, BoxMode, Keypoints, PolygonMasks, RotatedBoxes

from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
import os
import numpy as np
import torch

# Global variables that hold the models
model = None
DATA_FIELDS = ['FRONT_IMAGE']
FILTERthreshold=0.2
classes=('vehicle', 'pedestrian', 'sign', 'cyclist')

aug = None

def initialize_model():
    """Initialize the global model variable to the pretrained EfficientDet.
    This assumes that the EfficientDet model has already been downloaded to a
    specific path, as done in the Dockerfile for this example.
    """
    #model_dir = '/Developer/MyRepo/mymodels/tfssdresnet50_1024_ckpt100k/saved_model'
    model_dir = '/Developer/MyRepo/mymodels/detectron2models'
    modelname = 'faster_rcnn_X_101_32x8d_FPN_3x'
    modelfilename = 'model_0819999.pth'
    #load saved model
    global model
    
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/"+modelname+".yaml" ))#faster_rcnn_X_101_32x8d_FPN_3x
    #self.cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))#faster_rcnn_X_101_32x8d_FPN_3x
    #cfg.merge_from_file('faster_rcnn_R_101_C4_3x.yaml')#Tridentnet
    #cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/retinanet_R_101_FPN_3x.yaml"))
    #cfg.DATASETS.TRAIN = ("myuav1_train",)
    #cfg.DATASETS.TEST = ("myuav1_val",)
    cfg.DATALOADER.NUM_WORKERS = 1 #2
    #cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")  # Let training initialize from model zoo
    cfg.MODEL.WEIGHTS = os.path.join(model_dir, modelfilename) #model_0159999.pth
    if os.path.isfile(cfg.MODEL.WEIGHTS) == False:
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/"+self.args.modelname+".yaml")  # Let training initialize from model zoo
    else:
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512#128   # faster, and good enough for this toy dataset (default: 512)
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = len(classes)  # Kitti has 9 classes (including donot care)
    #self.cfg.MODEL.WEIGHTS = os.path.join('/home/010796032/PytorchWork/output_uav', "model_0119999.pth") #model_0159999.pth
    #cfg.MODEL.WEIGHTS = os.path.join('/home/010796032/PytorchWork', "fasterrcnn_x101_fpn_model_final_68b088.pkl")#using the local downloaded model
    cfg.SOLVER.IMS_PER_BATCH = 2
    cfg.SOLVER.BASE_LR = 0.00025  # pick a good LRgkyjmh,
    cfg.SOLVER.MAX_ITER = 100000    # you may need to train longer for a practical dataset

    cfg.TEST.DETECTIONS_PER_IMAGE = 50 #500
    
    #https://github.com/facebookresearch/detectron2/blob/master/detectron2/engine/defaults.py
    #model = DefaultPredictor(cfg) #runs on single device for a single input image
    model = build_model(cfg)
    model.eval()
    checkpointer = DetectionCheckpointer(model)
    checkpointer.load(cfg.MODEL.WEIGHTS)
    print(cfg.INPUT.FORMAT) #BGR, ["RGB", "BGR"]

    print(cfg.INPUT.MIN_SIZE_TEST)#800
    print(cfg.INPUT.MAX_SIZE_TEST)#1333
    global aug
    aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

def mypredictor(Imageinput):
    with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
        # Apply pre-processing to image.
        #if self.input_format == "RGB":
        # whether the model expects BGR inputs or RGB
        original_image = Imageinput[:, :, ::-1]
        # The : means "take everything in this dimension" and the ::-1 means "take everything in this dimension but backwards
        # you're flipping the color from RGB to BGR

        height, width = original_image.shape[:2]#1280, 1920
        image = aug.get_transform(original_image).apply_image(original_image)#800, 1200, 3
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))#3, 800, 1200

        inputs = {"image": image, "height": height, "width": width}
        predictions = model([inputs])[0]
        return predictions

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
    print(FRONT_IMAGE.size)
    imageshape=FRONT_IMAGE.shape
    im_width=imageshape[1]#1920
    im_height=imageshape[0]#1280
    #input_tensor = np.expand_dims(FRONT_IMAGE, 0)
    #detections = model(input_tensor)

    #outputs = model(FRONT_IMAGE)#use DefaultPredictor
    outputs = mypredictor(FRONT_IMAGE)

    pred_boxes = outputs["instances"].pred_boxes.tensor.cpu().numpy()
    pred_class = outputs["instances"].pred_classes.cpu().numpy()
    pred_score = outputs["instances"].scores.cpu().numpy()

    #(xmin, ymin), (xmax, ymax)
    #pred_boxes = [[(i[0], i[1]), (i[2], i[3])] for i in list(pred_boxes)] # Bounding boxes

    num_box=len(pred_boxes)
    outputboxes=[]
    scores=[]
    classes=[]
    if num_box<1:
        return {
            'boxes': [],
            'scores': [],
            'classes': [],
        }
    else:
        for index_i in range(num_box):
            if pred_score[index_i]>FILTERthreshold:
                center_x=(pred_boxes[index_i, 0] + pred_boxes[index_i, 2]) / 2.0 # (xmin+xmax)/2
                center_y=(pred_boxes[index_i, 1] + pred_boxes[index_i, 3])  / 2.0 # (ymin+ymax)/2
                width=(pred_boxes[index_i, 2] - pred_boxes[index_i, 0])
                height=(pred_boxes[index_i, 3] - pred_boxes[index_i, 1])
                outputboxes.append([center_x, center_y, width, height])#bboxes[index_i,0:3])
                scores.append(pred_score[index_i])
                classes.append(int(pred_class[index_i]+1))
        return {
            'boxes': np.array(outputboxes),
            'scores': np.array(scores),
            'classes': np.array(classes).astype(np.uint8),
        }