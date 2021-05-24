import matplotlib
import matplotlib.pyplot as plt

import os
import random
import io
import imageio
import glob
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont
from IPython.display import display, Javascript
from IPython.display import Image as IPyImage

import tensorflow as tf

from object_detection.utils import label_map_util
#from object_detection.utils import config_util
#from object_detection.utils import visualization_utils as viz_utils
#from object_detection.utils import colab_utils
from object_detection.builders import model_builder

#from MyDetector.Postprocess import postfilter

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

#%matplotlib inline
def load_image_into_numpy_array(path):
    """Load an image from file into a numpy array.
    Puts image into numpy array to feed into tensorflow graph.
    Note that by convention we put it into a numpy array with shape
    (height, width, channels), where channels=3 for RGB.
    Args:
    path: the file path to the image
    Returns:
    uint8 numpy array with shape (img_height, img_width, 3)
    """
    img_data = tf.io.gfile.GFile(path, 'rb').read()
    image = Image.open(BytesIO(img_data))
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape(
          (im_height, im_width, 3)).astype(np.uint8)

class MyTF2Detector(object):
    #args.showfig
    #args.modelname
    #args.modelbasefolder
    #args.modelfilename
    #args.labelmappath

    def __init__(self, args):
        self.args = args
        #self.FULL_LABEL_CLASSES=args.FULL_LABEL_CLASSES
        #self.threshold = args.threshold
        self.threshold = args.threshold if args.threshold is not None else 0.1
        
        tf.keras.backend.clear_session()
        self.detect_fn = tf.saved_model.load(args.modelbasefolder)
        
        label_map_path=args.labelmappath #'./models/research/object_detection/data/mscoco_label_map.pbtxt'
        label_map = label_map_util.load_labelmap(label_map_path)
        categories = label_map_util.convert_label_map_to_categories(
            label_map,
            max_num_classes=label_map_util.get_max_label_map_index(label_map),
            use_display_name=True)
        self.category_index = label_map_util.create_category_index(categories)
        label_map_dict = label_map_util.get_label_map_dict(label_map, use_display_name=True)
        self.FULL_LABEL_CLASSES=list(label_map_dict.keys())
        

    def detect(self, im):
        imageshape=im.shape
        im_width=imageshape[1]#2720#800
        im_height=imageshape[0]#1530#600
    
        input_tensor = np.expand_dims(im, 0)
        detections = self.detect_fn(input_tensor)
        
        #[0] means get the first batch, only one batch, 
        pred_boxes = detections['detection_boxes'][0].numpy() #xyxy type [0.26331702, 0.20630795, 0.3134004 , 0.2257505 ], [ymin, xmin, ymax, xmax]
        #print(boxes)
        pred_class = detections['detection_classes'][0].numpy().astype(np.uint8)#.astype(np.int32)
        #print(classes)
        pred_score = detections['detection_scores'][0].numpy() #decreasing order
        
        #Post filter based on threshold
        pred_boxes, pred_class, pred_score = postfilter(pred_boxes, pred_class, pred_score, self.threshold)
        if len(pred_class)>0:
            pred_class = [i-1 for i in list(pred_class)] # index starts with 1, 0 is the background in the tensorflow
            #normalized [ymin, xmin, ymax, xmax] to [ (xmin, ymin), (xmax, ymax)] in image size
            #pred_boxes = [[(i[1]*im_width, i[0]*im_height), (i[3]*im_width, i[2]*im_height)] for i in list(pred_boxes)] # Bounding boxes
            boxes = np.zeros_like(pred_boxes)
            boxes[:, 0] = (pred_boxes[:, 3] + pred_boxes[:, 1]) * im_width / 2.0
            boxes[:, 1] = (pred_boxes[:, 2] + pred_boxes[:, 0]) * im_height / 2.0
            boxes[:, 2] = (pred_boxes[:, 3] - pred_boxes[:, 1]) * im_width
            boxes[:, 3] = (pred_boxes[:, 2] - pred_boxes[:, 0]) * im_height

            #pred_boxes = [[(i[1]*im_width, i[0]*im_height), (i[3]*im_width, i[2]*im_height)] for i in list(pred_boxes)] # Bounding boxes in # (center_x, center_y, width, height) in pixels.
        resultdict= {
            'boxes': boxes,
            'scores': pred_score,
            'classes': pred_class,
        }
        return resultdict
        

        # #predlist=[scores.index(x) for x in scores if x > self.threshold] # Get list of index with score greater than threshold.
        # pred_score=[x for x in scores if x > self.threshold] # Get list of index with score greater than threshold.
        # #print(pred_score)
        # if len(pred_score)<1:
        #     print("Empty")
        #     pred_boxes=[]
        #     pred_class=[]
        #     pred_score=[]
        # else:
        #     pred_t = np.where(scores==pred_score[-1])#get the last index
        #     #print(pred_t)
        #     pred_t=pred_t[0][0] #fetch value from tuple of array
        #     #print(pred_t)
        #     print("Box len:", len(boxes))
        #     pred_boxes = boxes[:pred_t+1]
        #     print("pred_score len:", len(pred_score))
        #     #print("pred_boxes len:", len(pred_boxes))
        #     pred_class = classes[:pred_t+1]
        #     pred_class = [i-1 for i in list(pred_class)] # index starts with 1, 0 is the background in the tensorflow
        #     #print(pred_class)
            
        #     #[ (xmin, ymin), (xmax, ymax)]
        #     pred_boxes = [[(i[1]*im_width, i[0]*im_height), (i[3]*im_width, i[2]*im_height)] for i in list(pred_boxes)] # Bounding boxes
        
        #return pred_boxes, pred_class, pred_score

#def postprocess()