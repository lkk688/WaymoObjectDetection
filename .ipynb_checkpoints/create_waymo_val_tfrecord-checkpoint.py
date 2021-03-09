r"""Convert raw COCO 2017 dataset to TFRecord.

Example usage:
    python create_coco_tf_record.py --logtostderr \
      --image_dir="${TRAIN_IMAGE_DIR}" \
      --image_info_file="${TRAIN_IMAGE_INFO_FILE}" \
      --object_annotations_file="${TRAIN_ANNOTATIONS_FILE}" \
      --caption_annotations_file="${CAPTION_ANNOTATIONS_FILE}" \
      --output_file_prefix="${OUTPUT_DIR/FILE_PREFIX}" \
      --num_shards=100
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import hashlib
import io
import json
import multiprocessing
import os

from absl import app
from absl import flags
from absl import logging
import numpy as np
import PIL.Image

from pycocotools import mask
import tensorflow.compat.v1 as tf
#from dataset import label_map_util
#from dataset import tfrecord_util
import datetime
from glob import glob

from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils
from waymo_open_dataset.utils import  frame_utils
from waymo_open_dataset import dataset_pb2 as open_dataset


waymo_label_map_dict = {
    'Unknown': 0,
    'Vehicles': 1,
    'Pedestrians': 2,
    'Cyclists': 3,
}

def loadWaymoValidationFrames(PATH):
    validation_folders = ["validation_0000","validation_0001"]
    data_files = [path for x in validation_folders for path in glob(os.path.join(PATH, x, "*.tfrecord"))]
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

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))




def _create_tf_record_from_waymo_annotations(PATH,
                                            output_path, label_map_dict,
                                            num_shards):
    print("Loading Waymo training frames...")
    now = datetime.datetime.now()
    print ("Current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))
    
    #waymoallframes=loadWaymoFrames(PATH)#'/data/cmpe295-liu/Waymo')
    print("Loading Waymo validation frames...")
    waymoallframes=loadWaymoValidationFrames(PATH)
    
    image_width = 1920
    image_height = 1280
    label_map = {0:0, 1: 1, 2:2, 4:3} #4 object types (remove sign)
    INSTANCE_CATEGORY_NAMES = [
        'Unknown', 'Vehicles', 'Pedestrians', 'Cyclists'
    ]
    
    now = datetime.datetime.now()
    print ("Current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))
    
    ann_json_dict = {
      'images': [],
      'type': 'instances',
      'annotations': [],
      'categories': []
    }
    
    #logging.info('writing to output path: %s', FLAGS.output_path)
    writers = [
        tf.python_io.TFRecordWriter(output_path + '-%05d-of-%05d.tfrecord' %
                                  (i, num_shards))
        for i in range(num_shards)
    ]
    
    for class_name, class_id in label_map_dict.items():
        cls = {'supercategory': 'none', 'id': class_id, 'name': class_name}
        ann_json_dict['categories'].append(cls)
    
    totallen=len(waymoallframes)
    frameidx=0
    for i, frame in enumerate(waymoallframes): #Total number of TFrecord files * (Each TFrecord file, 40 frames)
        if i % 100 == 0:
            print('On image %d of %d, shard: %d', i, totallen, num_shards)
        
        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []
        classes = []
        classes_text = []
#         truncated = []
#         poses = []
#         difficult_obj = []
        filename=frame.context.name.split("_")[-2] + str(frameidx)
        #frame.context.name.split("_")[-2] + str(frame.timestamp_micros)
        #source_id= frame.context.name + str(frame.timestamp_micros)
        source_id= int(frame.context.name.split("_")[-2] + str(frameidx))
        frameidx=frameidx+1

        if ann_json_dict:
            imagejson = {
                'file_name': filename,
                'height': image_height,
                'width': image_width,
                'id': source_id,
            }
            ann_json_dict['images'].append(imagejson)
            
        
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
                if xmin<=xmax and ymin<=ymax and xmin>=0 and ymin>=0 and xmax<=image_width and ymax<=image_height:# and area>2000:
                    xmins.append(xmin/image_width) #normalized
                    ymins.append(ymin/image_height) #normalized
                    xmaxs.append(xmax/image_width)
                    ymaxs.append(ymax/image_height)

                    classid=label_map[label.type]
                    classes.append(classid) #int type
                    classes_text.append(INSTANCE_CATEGORY_NAMES[classid].encode('utf8'))
                    
                    if ann_json_dict:
                        abs_xmin = int(xmin)
                        abs_ymin = int(ymin)
                        abs_xmax = int(xmax)
                        abs_ymax = int(ymax)
                        abs_width = label.box.length #abs_xmax - abs_xmin
                        abs_height = label.box.width #abs_ymax - abs_ymin
                        ann = {
                            'area': area,
                            'iscrowd': 0,
                            'image_id': source_id, #image_id,
                            'bbox': [abs_xmin, abs_ymin, abs_width, abs_height],
                            'category_id': classid,
                            'id': frameidx,
                            'ignore': 0,
                        }
                        ann_json_dict['annotations'].append(ann)
                   
            if len(xmins)<=0: #drop frame
                continue
            #Decode a JPEG-encoded image to a uint8 tensor.
            encoded_jpg = frame.images[0].image
            #image = tf.image.decode_jpeg(frame.images[0].image).numpy()#front camera image
            
            
            
            tf_example = tf.train.Example(
                  features=tf.train.Features(
                      feature={
                          'image/height':
                              int64_feature(image_height),
                          'image/width':
                              int64_feature(image_width),
                          'image/filename':
                              bytes_feature(filename.encode('utf8')),
                          'image/source_id':
                              bytes_feature(str(source_id).encode('utf8')),
                          'image/encoded':
                              bytes_feature(encoded_jpg),
                          'image/format':
                              bytes_feature('jpeg'.encode('utf8')),
                          'image/object/bbox/xmin':
                              float_list_feature(xmins),
                          'image/object/bbox/xmax':
                              float_list_feature(xmaxs),
                          'image/object/bbox/ymin':
                              float_list_feature(ymins),
                          'image/object/bbox/ymax':
                              float_list_feature(ymaxs),
                          'image/object/class/text':
                              bytes_list_feature(classes_text),
                          'image/object/class/label':
                              int64_list_feature(classes),
              }))
            writers[i % num_shards].write(tf_example.SerializeToString())
            
    
    for writer in writers:
        writer.close()
    
    json_file_path = os.path.join(
        os.path.dirname(output_path), 'json_' + os.path.basename(output_path) + '.json')
    with tf.io.gfile.GFile(json_file_path, 'w') as f:
        json.dump(ann_json_dict, f)
    
    now = datetime.datetime.now()
    print ("Finished. Current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))
                

def main(_):
    
    num_shards=5
    label_map_dict = waymo_label_map_dict
    _create_tf_record_from_waymo_annotations('/data/cmpe295-liu/Waymo',
                                            '/data/cmpe295-liu/Waymo/NewTFRecord/TFRecordVal-',
                                             label_map_dict,
                                            num_shards)


if __name__ == '__main__':
    app.run(main)



