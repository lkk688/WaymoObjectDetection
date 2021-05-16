import tensorflow as tf
print("GPU Available: ", tf.test.is_gpu_available())

print("Tensorflow Version: ", tf.__version__)
print("Keras Version: ", tf.keras.__version__)

#check GPU
physical_devices = tf.config.experimental.list_physical_devices('GPU')
print(physical_devices)

import matplotlib
import matplotlib.pyplot as plt

import io
import scipy.misc
import numpy as np
from six import BytesIO
from PIL import Image, ImageDraw, ImageFont

import tensorflow as tf

import pandas as pd
import os, sys, math
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as patches
import cv2

from object_detection.utils import label_map_util
from object_detection.utils import config_util
from object_detection.utils import visualization_utils as viz_utils
from object_detection.builders import model_builder

#ref: https://github.com/tensorflow/models/blob/master/official/vision/detection/dataloader/tf_example_decoder.py
def _decode_image(parsed_tensors):
    """Decodes the image and set its static shape."""
    image = tf.io.decode_image(parsed_tensors['image/encoded'], channels=3)
    image.set_shape([None, None, 3])
    return image

def _decode_boxes(parsed_tensors):
    """Concat box coordinates in the format of [ymin, xmin, ymax, xmax]."""
    xmin = parsed_tensors['image/object/bbox/xmin']
    xmax = parsed_tensors['image/object/bbox/xmax']
    ymin = parsed_tensors['image/object/bbox/ymin']
    ymax = parsed_tensors['image/object/bbox/ymax']
    print(ymax)

    return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

def _decode_areas(parsed_tensors):
    xmin = parsed_tensors['image/object/bbox/xmin']
    xmax = parsed_tensors['image/object/bbox/xmax']
    ymin = parsed_tensors['image/object/bbox/ymin']
    ymax = parsed_tensors['image/object/bbox/ymax']
    return tf.cond(
        tf.greater(tf.shape(parsed_tensors['image/object/area'])[0], 0),
        lambda: parsed_tensors['image/object/area'],
        lambda: (xmax - xmin) * (ymax - ymin))

#classlabelkeyname='image/object/class/label' #used in the previous TF record file
classlabelkeyname='image/object/class/text'#used in the new TF record file
# 'image/object/class/text':
#             tf.io.VarLenFeature(tf.int64),
def read_tfrecord(example):
    features = {
        'image/encoded':
            tf.io.FixedLenFeature((), tf.string),
        'image/source_id':
            tf.io.FixedLenFeature((), tf.string),
        'image/height':
            tf.io.FixedLenFeature((), tf.int64),
        'image/width':
            tf.io.FixedLenFeature((), tf.int64),
        'image/object/bbox/xmin':
            tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/xmax':
            tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymin':
            tf.io.VarLenFeature(tf.float32),
        'image/object/bbox/ymax':
            tf.io.VarLenFeature(tf.float32),
        'image/object/class/label':
            tf.io.VarLenFeature(tf.int64),
        'image/object/class/text':
            tf.io.VarLenFeature(tf.string),
        'image/object/area':
            tf.io.VarLenFeature(tf.float32),
        'image/object/is_crowd':
            tf.io.VarLenFeature(tf.int64),
    }
    # features = {
    #     "image": tf.io.FixedLenFeature([], tf.string), # tf.string means bytestring
    #     "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means scalar
    # }
    example = tf.io.parse_single_example(example, features)

    for k in example:
      if isinstance(example[k], tf.SparseTensor):
        if example[k].dtype == tf.string:
          example[k] = tf.sparse.to_dense(
              example[k], default_value='')
        else:
          example[k] = tf.sparse.to_dense(
              example[k], default_value=0)
          
    print("Got example")
    print(example['image/object/bbox/xmin'])
    image = _decode_image(example)
    print("Decoded image")
    boxes = _decode_boxes(example)
    print("Decoded boxes:", boxes)
    areas = _decode_areas(example)
    is_crowds = tf.cond(
        tf.greater(tf.shape(example['image/object/is_crowd'])[0], 0),
        lambda: tf.cast(example['image/object/is_crowd'], dtype=tf.bool),
        lambda: tf.zeros_like(example['image/object/class/label'], dtype=tf.bool))  # pylint: disable=line-too-long

    source_id = example['image/source_id']
    height=example['image/height']
    width=example['image/width']
    groundtruth_class=example['image/object/class/label']#['image/object/class/label']

    #image = tf.image.decode_jpeg(example['image'], channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.image.resize(image, IMAGE_SIZE)
    #image = tf.reshape(image, [*IMAGE_SIZE, 3]) # explicit size will be needed for TPU
    #class_label = example['class']
    print(groundtruth_class)
    decoded_tensors = {
        'image': image,
        'source_id': source_id,
        'height':height,
        'width':width,
        'groundtruth_classes': groundtruth_class,
        'groundtruth_is_crowd': is_crowds,
        'groundtruth_area': areas,
        'groundtruth_boxes': boxes,
    }
    return decoded_tensors#image, class_label

def load_dataset(filenames):
    # read from TFRecords. For optimal performance, read from multiple
    # TFRecord files at once and set the option experimental_deterministic = False
    # to allow order-altering optimizations.

    option_no_order = tf.data.Options()
    option_no_order.experimental_deterministic = False

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO)
    dataset = dataset.with_options(option_no_order)
    dataset = dataset.map(read_tfrecord, num_parallel_calls=AUTO)
    return dataset

INSTANCE_CATEGORY_NAMES =['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']
# INSTANCE_Color = {
#     'Unknown':'black', b'vehicle':'red', b'pedestrian':'green', b'sign':'red', b'cyclist':'purple'
# }#'Unknown', 'Vehicles', 'Pedestrians', 'Cyclists'
INSTANCE_Color = ['black', 'red', 'green', 'red', 'purple']


def show_oneimage_category(image, label, boundingbox, IMAGE_SIZE):
    fig = plt.figure(figsize=(30,30))
    ax = fig.add_subplot(1, 1, 1)

    len=label.size
    print(len)
    img_height=IMAGE_SIZE[0]
    img_width=IMAGE_SIZE[1]
    for index in range(len):
        box= boundingbox[index]
        labelid=label[index]
        print(labelid)
        labelname=INSTANCE_CATEGORY_NAMES[labelid] #labelid #INSTANCE_CATEGORY_NAMES[labelid]
        classcolor=INSTANCE_Color[int(labelid)]#labelname]
        #[xmin, ymin, xmax, ymax]=box#*IMAGE_SIZE[0]
        [ymin, xmin, ymax, xmax]=box
        xmin=xmin*img_width
        xmax=xmax*img_width
        ymin=ymin*img_height
        ymax=ymax*img_height
        boxwidth=xmax-xmin
        boxheight=ymax-ymin
        if (boxwidth/img_width>0.01 and boxheight/img_height>0.01):
            print("Class id:", labelid)
            print(box)
            startpoint = (xmin, ymin)
            end_point = (xmax, ymax)
            #cv2.rectangle(image, startpoint, end_point, color=(0, 255, 0), thickness=1) # Draw Rectangle with the coordinates
            # Draw the object bounding box. https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.Rectangle.html
            ax.add_patch(patches.Rectangle(
                xy=(xmin,
                    ymin),
                width=boxwidth,
                height=boxheight,
                linewidth=1,
                edgecolor=classcolor, #'red',
                facecolor='none'))
            # ax.add_patch(patches.Rectangle(
            #     xy=(ymin,
            #         xmin),
            #     width=ymax-ymin,
            #     height=xmax-xmin,
            #     linewidth=1,
            #     edgecolor=classcolor, #'red',
            #     facecolor='none'))
            #ax.text(ymin, xmin, labelname, color=classcolor, fontsize=10)
            ax.text(xmin, ymin, labelname, color=classcolor, fontsize=10)
            text_size = 1
            #cv2.putText(image, labelname, startpoint,  cv2.FONT_HERSHEY_SIMPLEX, text_size, (0,255,0),thickness=1)
            
    #return img
    #plt.imshow(image)
    plt.imsave('test.png', image)
    #plt.title(CLASSES[label_batch.numpy()])

if __name__ == "__main__":
    
    AUTO = tf.data.experimental.AUTOTUNE
    #Original image size width=1920, height=1280
    IMAGE_SIZE = [1280, 1920] #[640, 640] #[192, 192]
    #TPU can only use data from Google Cloud Storage
    train_filenames = tf.io.gfile.glob('/DATA5T/Dataset/WaymoTFRecord/train100val20/TFRecordValBig--00000-of-00005.tfrecord')#TPU can only load data from google cloud
    display_dataset=load_dataset(train_filenames)
    display_dataset_iter=iter(display_dataset)
    decoded_tensors =next(display_dataset_iter)

    #decoded_tensors =next(iter(display_dataset))
    print(decoded_tensors['groundtruth_boxes'].numpy())
    print("Image width:", decoded_tensors['width'].numpy())
    print("Image height:", decoded_tensors['height'].numpy())
    print("Groundtruth classes:", decoded_tensors['groundtruth_classes'].numpy())

    testimage=decoded_tensors['image']
    testlabel=decoded_tensors['groundtruth_classes'].numpy()
    testboundingbox=decoded_tensors['groundtruth_boxes'].numpy()
    show_oneimage_category(testimage, testlabel, testboundingbox, IMAGE_SIZE)
    #cv2.imwrite('result.jpg', resultimage) 

