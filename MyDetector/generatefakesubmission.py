import tensorflow.compat.v1 as tf
from pathlib import Path
import json
import argparse
import tqdm
import uuid

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import label_pb2
from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset.protos import submission_pb2

def _fancy_deep_learning(frame):
    """Creates a prediction objects file."""
    o_list = []

    for camera_labels in frame.camera_labels:
        if camera_labels.name != 1: #Only use front camera
            continue
        for gt_label in camera_labels.labels:
            o = metrics_pb2.Object()
            # The following 3 fields are used to uniquely identify a frame a prediction
            # is predicted at.
            o.context_name = frame.context.name
            # The frame timestamp for the prediction. See Frame::timestamp_micros in
            # dataset.proto.
            o.frame_timestamp_micros = frame.timestamp_micros
            # This is only needed for 2D detection or tracking tasks.
            # Set it to the camera name the prediction is for.
            o.camera_name = camera_labels.name

            # Populating box and score.
            box = label_pb2.Label.Box()
            box.center_x = gt_label.box.center_x
            box.center_y = gt_label.box.center_y
            box.length =   gt_label.box.length
            box.width =    gt_label.box.width
            o.object.box.CopyFrom(box)
            # This must be within [0.0, 1.0]. It is better to filter those boxes with
            # small scores to speed up metrics computation.
            o.score = 0.9
            # Use correct type.
            o.object.type = gt_label.type
            o_list.append(o)

    return o_list

from glob import glob
import os
if __name__ == "__main__":
    PATH='/data/cmpe295-liu/Waymo'
    #validation_folders = ["validation_0000"]
    validation_folders = ["validation_0000","validation_0001","validation_0002","validation_0003","validation_0004","validation_0005","validation_0006","validation_0007"] #["validation_0007","validation_0006","validation_0005","validation_0004","validation_0003","validation_0002","validation_0001","validation_0000"]
    data_files = [path for x in validation_folders for path in glob(os.path.join(PATH, x, "*.tfrecord"))]
    print(data_files)#all TFRecord file list
    print(len(data_files))
    #dataset = tf.data.TFRecordDataset([str(x.absolute()) for x in Path(data_files)])
    dataset = [tf.data.TFRecordDataset(FILENAME, compression_type='') for FILENAME in data_files]#create a list of dataset for each TFRecord file
    print("Dataset type:",type(dataset))
    frames = [] #store all frames = total number of TFrecord files * 40 frame(each TFrecord)
    objects = metrics_pb2.Objects()
    for i, data_file in enumerate(dataset):
        print("Datafile: ",i)#Each TFrecord file
        for idx, data in enumerate(data_file): #Create frame based on Waymo API, 199 frames per TFrecord (20s, 10Hz)
#             if idx % 5 != 0: #Downsample every 5 images, reduce to 2Hz, total around 40 frames
#                 continue
            frame = dataset_pb2.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            o_list = _fancy_deep_learning(frame)
            frames.append(frame)
            for o in o_list:
                objects.objects.append(o)

    #https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/protos/submission.proto
    submission = submission_pb2.Submission()
    submission.task = submission_pb2.Submission.DETECTION_2D 
    submission.account_name = 'kaikai.liu@sjsu.edu'
    submission.authors.append('Kaikai Liu')
    submission.affiliation = 'None'
    submission.unique_method_name = 'fake'
    submission.description = 'none'
    submission.method_link = "empty method"
    submission.sensor_type = submission_pb2.Submission.CAMERA_ALL
    submission.number_past_frames_exclude_current = 0
    submission.number_future_frames_exclude_current = 0
    submission.inference_results.CopyFrom(objects)
    submission.docker_image_source = '' #// Link to the latency submission Docker image stored in Google Storage bucket
    #object_types // Object types this submission contains. By default, we assume all types.
    #latency_second Self-reported end to end inference latency in seconds
    
    outputfilepath='/home/010796032/MyRepo/submissionoutput/fake_valfrontcameraall.bin'
    f = open(outputfilepath, 'wb')
    f.write(submission.SerializeToString())
    f.close()