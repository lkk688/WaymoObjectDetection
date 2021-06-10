"""Module to load and run an Tensorflow model."""
import os
#from . import TF2Detector

import numpy as np
from object_detection.builders import model_builder
from object_detection.utils import config_util
import tensorflow as tf



# Global variables that hold the models
model = None
DATA_FIELDS = ['FRONT_IMAGE']
FILTERthreshold=0.2
model_dir = '/Developer/MyRepo/mymodels/tf_ssdresnet50_output/exported130/saved_model'
config = ''#Use optional load from checkpoint
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
    #model_dir = '/Developer/MyRepo/mymodels/tfssdresnet50_1024_ckpt100k/saved_model'
    #model_dir = '/Developer/MyRepo/mymodels/tf_ssdresnet50_output/exported130/saved_model'
    #load saved model
    global model
    model = tf.saved_model.load(model_dir)
    print(model.signatures['serving_default'].inputs)
    print(model.signatures['serving_default'].output_shapes)

    # configs = config_util.get_configs_from_pipeline_file(
    #     os.path.join(model_dir, 'pipeline.config'))
    # model_config = configs['model']
    # global model
    # model = model_builder.build(model_config=model_config, is_training=False)
    # ckpt = tf.train.Checkpoint(model=model)
    # ckpt.restore(os.path.join(model_dir, 'checkpoint', 'ckpt-0'))


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
    #print(FRONT_IMAGE.size)
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
            'boxes':  np.array(boxes),
            'scores': np.array(pred_score),
            'classes': np.array(pred_class).astype(np.uint8),
        }
    else:#empty
        return {
            'boxes': np.array(pred_boxes),
            'scores': np.array(pred_score),
            'classes': np.array(pred_class).astype(np.uint8),
        }

    # inp_tensor = tf.convert_to_tensor(
    #     np.expand_dims(FRONT_IMAGE, 0), dtype=tf.float32)
    # image, shapes = model.preprocess(inp_tensor)
    # pred_dict = model.predict(image, shapes)
    # detections = model.postprocess(pred_dict, shapes)
    # corners = detections['detection_boxes'][0, ...].numpy()
    # scores = detections['detection_scores'][0, ...].numpy()
    # coco_classes = detections['detection_classes'][0, ...].numpy()
    # coco_classes = coco_classes.astype(np.uint8)

    # # Convert the classes from COCO classes to WOD classes, and only keep
    # # detections that belong to a WOD class.
    # coco_classes = [i+1 for i in list(coco_classes)]
    # wod_classes = np.vectorize(translate_label_to_wod)(coco_classes)
    # corners = corners[wod_classes > 0]
    # scores = scores[wod_classes > 0]
    # classes = wod_classes[wod_classes > 0]
    # # Note: the boxes returned by the TF OD API's pretrained models returns boxes
    # # in the format (ymin, xmin, ymax, xmax), normalized to [0, 1]. Thus, this
    # # function needs to convert them to the format expected by WOD, namely
    # # (center_x, center_y, width, height) in pixels.
    # boxes = np.zeros_like(corners)
    # # boxes[:, 0] = (corners[:, 3] + corners[:, 1]) * FRONT_IMAGE.shape[0] / 2.0 #(1280, 1920, 3)
    # # boxes[:, 1] = (corners[:, 2] + corners[:, 0]) * FRONT_IMAGE.shape[1] / 2.0
    # # boxes[:, 2] = (corners[:, 3] - corners[:, 1]) * FRONT_IMAGE.shape[0] #width
    # # boxes[:, 3] = (corners[:, 2] - corners[:, 0]) * FRONT_IMAGE.shape[1] # height

    # height=FRONT_IMAGE.shape[0]
    # width=FRONT_IMAGE.shape[1]
    # boxes[:, 0] = (corners[:, 3] + corners[:, 1]) * width / 2.0 #center_x
    # boxes[:, 1] = (corners[:, 2] + corners[:, 0]) * height / 2.0
    # boxes[:, 2] = (corners[:, 3] - corners[:, 1]) * width #width
    # boxes[:, 3] = (corners[:, 2] - corners[:, 0]) * height # height

    # return {
    #     'boxes': boxes,
    #     'scores': scores,
    #     'classes': classes,
    # }
# BEGIN-INTERNAL
# pylint: disable=invalid-name
# END-INTERNAL
