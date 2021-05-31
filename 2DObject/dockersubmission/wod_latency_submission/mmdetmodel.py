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
classes=('vehicle', 'pedestrian', 'sign', 'cyclist')

model_dir = '/Developer/MyRepo/mymodels/mmmodels/HPCwaymococo_fasterrcnnr101train/epoch_25.pth'
Basemmdetection='/Developer/3DObject/mmdetection/'
config = Basemmdetection+'configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py'

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
    
    # Setup a checkpoint file to load
    #checkpoint = '/Developer/3DObject/mymmdetection/waymococo_fasterrcnnr101train/epoch_60.pth'
    
    #checkpoint = '/Developer/MyRepo/mymodels/mmmodels/HPCwaymococo_fasterrcnnr101train/epoch_25.pth'
    device='cuda:0'
    global model
    global config
    global model_dir
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    config.model.train_cfg = None
    config.model.roi_head.bbox_head.num_classes = len(classes)
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if model_dir is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, model_dir, map_location=map_loc)
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
#REF: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/apis/inference.py
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
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
    im_width=imageshape[1]#1920
    im_height=imageshape[0]#1280
    
    datas = []

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    cfg = cfg.copy()
    # set loading pipeline type
    cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    if isinstance(FRONT_IMAGE, np.ndarray):
        # directly add img
        data = dict(img=FRONT_IMAGE)
    # build the data pipeline
    data = test_pipeline(data)
    datas.append(data)

    data = collate(datas, samples_per_gpu=1)
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]

    data = scatter(data, [device])[0]
    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)
    #return results[0]
    bbox_result=results[0]
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    #pred_class = [i+1 for i in list(labels)] #output labels starts with index =0
    #print(bboxes)#5,1, last column is the confidence score
    #xmin, ymin, xmax, ymax in pixels
    #print(type(bboxes))
    #print(type(labels))

    num_box=len(bboxes)
    outputboxes=[]
    scores=[]
    classes=[]
    if num_box<1:
        return {
            'boxes': np.array(outputboxes),
            'scores': np.array(scores),
            'classes': np.array(classes).astype(np.uint8),
        }
    else:
        for index_i in range(num_box):
            if bboxes[index_i,4]>FILTERthreshold:
                center_x=(bboxes[index_i, 0] + bboxes[index_i, 2]) / 2.0
                center_y=(bboxes[index_i, 1] + bboxes[index_i, 3])  / 2.0
                width=(bboxes[index_i, 2] - bboxes[index_i, 0])
                height=(bboxes[index_i, 3] - bboxes[index_i, 1])
                outputboxes.append([center_x, center_y, width, height])#bboxes[index_i,0:3])
                scores.append(bboxes[index_i,4])
                classes.append(int(labels[index_i]+1))
        return {
            'boxes': np.array(outputboxes),
            'scores': np.array(scores),
            'classes': np.array(classes).astype(np.uint8),
        }
    # newboxes = np.zeros((num_box,4), dtype=float)
    # newboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2.0 #center_x
    # newboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3])  / 2.0 #center_y
    # newboxes[:, 2] = (bboxes[:, 2] - bboxes[:, 0]) #width
    # newboxes[:, 3] = (bboxes[:, 3] - bboxes[:, 1]) # height
    # # boxes[:, 0] = (pred_boxes[:, 3] + pred_boxes[:, 1]) * im_width / 2.0
    # # boxes[:, 1] = (pred_boxes[:, 2] + pred_boxes[:, 0]) * im_height / 2.0
    # # boxes[:, 2] = (pred_boxes[:, 3] - pred_boxes[:, 1]) * im_width
    # # boxes[:, 3] = (pred_boxes[:, 2] - pred_boxes[:, 0]) * im_height
    # pred_score = np.zeros((num_box,), dtype=float)
    # pred_score = bboxes[:,4]
    # newboxes, pred_class, pred_score = postfilter(newboxes, pred_class, pred_score, FILTERthreshold)
    # if len(pred_score)!=len(newboxes):
    #     print("error")
    # return {
    #     'boxes': newboxes,
    #     'scores': pred_score,
    #     'classes': pred_class,
    # }

