#ref: https://github.com/tensorflow/models/blob/master/research/object_detection/dataset_tools/create_coco_tf_record.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import hashlib
import io
import json
import logging
import os
#import contextlib2
import numpy as np
import PIL.Image
#import tensorflow.compat.v1 as tf
#import tensorflow as tf
import tensorflow.compat.v1 as tf

#from dataset import label_map_util
#from dataset import tfrecord_util
import datetime
from glob import glob

# waymo_label_map_dict = {
#     'Unknown': 0,
#     'Vehicles': 1,
#     'Pedestrians': 2,
#     'Cyclists': 3,
# }

def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))

def create_category_index(categories):
    """Creates dictionary of COCO compatible categories keyed by category id.
    Args:
    categories: a list of dicts, each of which has the following keys:
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name
        e.g., 'cat', 'dog', 'pizza'.
    Returns:
    category_index: a dict containing the same entries as categories, but keyed
        by the 'id' field of each category.
    """
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index

def create_tf_example(image,
                      annotations_list,
                      image_dir,
                      category_index,
                      include_masks=False,
                      keypoint_annotations_dict=None,
                      densepose_annotations_dict=None,
                      remove_non_person_annotations=False,
                      remove_non_person_images=False):
    image_height = image['height']#1280
    image_width = image['width']#1920
    filename = image['file_name']
    print("Image filename:", filename)
    image_id = image['id']
    full_path = os.path.join(image_dir, filename)
    with tf.gfile.GFile(full_path, 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = PIL.Image.open(encoded_jpg_io)
    key = hashlib.sha256(encoded_jpg).hexdigest()

    xmin = []
    xmax = []
    ymin = []
    ymax = []
    is_crowd = []
    category_names = []
    category_ids = []
    area = []
    num_annotations_skipped=0
    images_noannotation=0

    for object_annotations in annotations_list:
        (x, y, width, height) = tuple(object_annotations['bbox'])
        if width <= 0 or height <= 0:
            num_annotations_skipped += 1
            continue
        if x + width > image_width or y + height > image_height:
            num_annotations_skipped += 1
            continue
        category_id = int(object_annotations['category_id'])
        category_name = category_index[category_id]['name'].encode('utf8')
        xmin.append(float(x) / image_width)
        xmax.append(float(x + width) / image_width)
        ymin.append(float(y) / image_height)
        ymax.append(float(y + height) / image_height)
        is_crowd.append(object_annotations['iscrowd'])
        category_ids.append(category_id)
        category_names.append(category_name)
        area.append(object_annotations['area'])
    
    if len(xmin)<1:
        images_noannotation += 1
        print("No Annotations, image id:", image_id)
    feature_dict = {
        'image/height':
            int64_feature(image_height),
        'image/width':
            int64_feature(image_width),
        'image/filename':
            bytes_feature(filename.encode('utf8')),
        'image/source_id':
            bytes_feature(str(image_id).encode('utf8')),
        'image/key/sha256':
            bytes_feature(key.encode('utf8')),
        'image/encoded':
            bytes_feature(encoded_jpg),
        'image/format':
            bytes_feature('jpeg'.encode('utf8')),
        'image/object/bbox/xmin':
            float_list_feature(xmin),
        'image/object/bbox/xmax':
            float_list_feature(xmax),
        'image/object/bbox/ymin':
            float_list_feature(ymin),
        'image/object/bbox/ymax':
            float_list_feature(ymax),
        'image/object/class/text':
            bytes_list_feature(category_names),
        'image/object/class/label':
            int64_list_feature(category_ids),
        'image/object/is_crowd':
            int64_list_feature(is_crowd),
        'image/object/area':
            float_list_feature(area),
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return (key, example, num_annotations_skipped, images_noannotation)


def _create_tf_record_from_coco_annotations(annotations_file, image_dir,
                                            output_path,
                                            num_shards):
    """Loads COCO annotation json files and converts to tf.Record format.
    Args:
    annotations_file: JSON file containing bounding box annotations.
    image_dir: Directory containing the image files.
    output_path: Path to output tf.Record file.
    num_shards: number of output file shards.
    """
    with open(annotations_file) as f:
        groundtruth_data = json.load(f)
    #groundtruth_data = json.load(annotations_file)
    images = groundtruth_data['images']
    print("Total images:",len(images))
    categories=groundtruth_data['categories']
    print("Categories:", categories)
    category_index = create_category_index(groundtruth_data['categories'])
    print("category_index:", category_index)

    writers = [
        tf.python_io.TFRecordWriter(output_path + '-%05d-of-%05d.tfrecord' %
                                  (i, num_shards))
        for i in range(num_shards)
    ]
    # writers = [
    #     tf.io.TFRecordWriter(output_path + '-%05d-of-%05d.tfrecord' %
    #                               (i, num_shards))
    #     for i in range(num_shards)
    # ]

    annotations_index = {}
    if 'annotations' in groundtruth_data:
        print('Found groundtruth annotations. Building annotations index.')
        print("Total annotations:", len(groundtruth_data['annotations']))
        for annotation in groundtruth_data['annotations']:
            image_id = annotation['image_id']
            if image_id not in annotations_index:
                annotations_index[image_id] = []
            annotations_index[image_id].append(annotation)
    print("Total annotations_index:", len(annotations_index))
    missing_annotation_count = 0
    for image in images:
        image_id = image['id']
        if image_id not in annotations_index:
            missing_annotation_count += 1
            annotations_index[image_id] = []
    print(f"{missing_annotation_count} images are missing annotations..")
    
    total_num_annotations_skipped = 0
    total_images_noannotation=0
    for idx, image in enumerate(images):
        if idx % 100 == 0:
            print(f"On image {idx} of {len(images)} images.")
        annotations_list = annotations_index[image['id']]
        # create_tf_example
        (_, tf_example, num_annotations_skipped, images_noannotation) = create_tf_example(image, annotations_list, image_dir, category_index)
        total_num_annotations_skipped += num_annotations_skipped
        total_images_noannotation += images_noannotation
        print(f"total_images_noannotation: {total_images_noannotation}")
        shard_idx = idx % num_shards
        if tf_example:
            #output_tfrecords[shard_idx].write(tf_example.SerializeToString())
            print("Write tf_example")
            writers[shard_idx].write(tf_example.SerializeToString())
    
    for writer in writers:
        writer.close()
    print(f'Finished writing, skipped {total_num_annotations_skipped} annotations.')
    

if __name__ == "__main__":
    num_shards=10
    #label_map_dict = waymo_label_map_dict
    #annotations_file = "/DATA5T/Dataset/WaymoCOCO/annotations_train200filteredbig.json"
    annotations_file = "/DATA5T/Dataset/WaymoCOCO/annotations_train100filteredbig.json"
    image_dir= '/DATA5T/Dataset/WaymoCOCO/'
    #output_path='/DATA5T/Dataset/WaymoTFRecord/TFRecordTrain-'
    output_path='/DATA5T/Dataset/WaymoTFRecord/train100val20/TFRecordTrainBig-'
    #_create_tf_record_from_coco_annotations(annotations_file, image_dir, output_path, num_shards)

    #annotations_file = "/DATA5T/Dataset/WaymoCOCO/annotations_val50filteredbig.json"
    #output_path='/DATA5T/Dataset/WaymoTFRecord/TFRecordVal-'
    annotations_file = "/DATA5T/Dataset/WaymoCOCO/annotations_val20filteredbig.json"
    output_path='/DATA5T/Dataset/WaymoTFRecord/train100val20/TFRecordValBig-' 
    num_shards=5
    #_create_tf_record_from_coco_annotations(annotations_file, image_dir, output_path, num_shards) 


    annotations_file = "/DATA5T/Dataset/WaymoCOCO/annotations_train684filteredbig.json"
    image_dir= '/DATA5T/Dataset/WaymoCOCO/'
    #output_path='/DATA5T/Dataset/WaymoTFRecord/TFRecordTrain-'
    output_path='/DATA5T/Dataset/WaymoTFRecord/trainvalall/TFRecordTrain684Big-'
    num_shards=40
    _create_tf_record_from_coco_annotations(annotations_file, image_dir, output_path, num_shards)

    #annotations_file = "/DATA5T/Dataset/WaymoCOCO/annotations_val50filteredbig.json"
    #output_path='/DATA5T/Dataset/WaymoTFRecord/TFRecordVal-'
    annotations_file = "/DATA5T/Dataset/WaymoCOCO/annotations_val202filteredbig.json"
    output_path='/DATA5T/Dataset/WaymoTFRecord/trainvalall/TFRecordVal202Big-' 
    num_shards=10
    _create_tf_record_from_coco_annotations(annotations_file, image_dir, output_path, num_shards)  