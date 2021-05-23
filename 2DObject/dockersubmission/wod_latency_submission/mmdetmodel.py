import torch
import os
import numpy as np
import torchvision
# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

from mmcv import Config
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes
import mmcv
import numpy as np

# Global variables that hold the models
model = None
DATA_FIELDS = ['FRONT_IMAGE']
FILTERthreshold=0.2

def initialize_model():
    Basemmdetection='/Developer/3DObject/mmdetection/'
    config = Basemmdetection+'configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py'
    # Setup a checkpoint file to load
    checkpoint = '/Developer/3DObject/mymmdetection/waymococo_fasterrcnnr101train/epoch_60.pth'
    device='cuda:0'
    global model
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    


def translate_label_to_wod(label):
    """Translate a single COCO class to its corresponding WOD class.
    Note: Returns -1 if this COCO class has no corresponding class in WOD.
    Args:
      label: int COCO class label
    Returns:
      Int WOD class label, or -1.
    """
    label_conversion_map = {
        1: 2,   # Person is ped
        2: 4,   # Bicycle is bicycle
        3: 1,   # Car is vehicle
        4: 1,   # Motorcycle is vehicle
        6: 1,   # Bus is vehicle
        8: 1,   # Truck is vehicle
        13: 3,  # Stop sign is sign
    }
    return label_conversion_map.get(label, -1)

def postfilter(boxes, classes, scores, threshold):
    pred_score=[x for x in scores if x > threshold] # Get list of score with score greater than threshold.
    #print(pred_score)
    if len(pred_score)<1:
        print("Empty")
        pred_boxes=[]
        pred_class=[]
        pred_score=[]
    else:
        pred_t = np.where(scores==pred_score[-1])#get the last index
        #print(pred_t)
        pred_t=pred_t[0][0] #fetch value from tuple of array, (array([2]),)
        #print(pred_t)
        print("Box len:", len(boxes))
        pred_boxes = boxes[:pred_t+1]
        print("pred_score len:", len(pred_score))
        #print("pred_boxes len:", len(pred_boxes))
        pred_class = classes[:pred_t+1]
    return pred_boxes, pred_class, pred_score


# BEGIN-INTERNAL
# pylint: disable=invalid-name
# END-INTERNAL
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
    input_tensor = np.expand_dims(FRONT_IMAGE, 0)
    detections = model(input_tensor)

    pred_boxes = detections['detection_boxes'][0].numpy() #xyxy type [0.26331702, 0.20630795, 0.3134004 , 0.2257505 ], [ymin, xmin, ymax, xmax]
    #print(boxes)
    pred_class = detections['detection_classes'][0].numpy().astype(np.uint8)#.astype(np.int32)
    #print(classes)
    pred_score = detections['detection_scores'][0].numpy() #decreasing order
    
    #Post filter based on threshold
    pred_boxes, pred_class, pred_score = postfilter(pred_boxes, pred_class, pred_score, FILTERthreshold)
    if len(pred_class)>0:
        #pred_class = [i-1 for i in list(pred_class)] # index starts with 1, 0 is the background in the tensorflow
        #normalized [ymin, xmin, ymax, xmax] to (center_x, center_y, width, height) in image size
        #pred_boxes = [[(i[1]*im_width, i[0]*im_height), (i[3]*im_width, i[2]*im_height)] for i in list(pred_boxes)] # Bounding boxes
        boxes = np.zeros_like(pred_boxes)
        boxes[:, 0] = (pred_boxes[:, 3] + pred_boxes[:, 1]) * im_width / 2.0 #center_x
        boxes[:, 1] = (pred_boxes[:, 2] + pred_boxes[:, 0]) * im_height / 2.0
        boxes[:, 2] = (pred_boxes[:, 3] - pred_boxes[:, 1]) * im_width #width
        boxes[:, 3] = (pred_boxes[:, 2] - pred_boxes[:, 0]) * im_height # height
        # boxes[:, 0] = (pred_boxes[:, 3] + pred_boxes[:, 1]) * im_width / 2.0
        # boxes[:, 1] = (pred_boxes[:, 2] + pred_boxes[:, 0]) * im_height / 2.0
        # boxes[:, 2] = (pred_boxes[:, 3] - pred_boxes[:, 1]) * im_width
        # boxes[:, 3] = (pred_boxes[:, 2] - pred_boxes[:, 0]) * im_height

        return {
            'boxes': boxes,
            'scores': pred_score,
            'classes': pred_class,
        }
    else:#empty
        return {
            'boxes': pred_boxes,
            'scores': pred_score,
            'classes': pred_class,
        }

