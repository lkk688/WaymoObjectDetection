#generated dict files in /home/010796032/MyRepo/WaymoObjectDetection/DatasetTools/waymoframe2dict.py
#dict file path: /data/cmpe249-f20/WaymoKittitMulti/validationalldicts
#test loading one dict file: /home/010796032/MyRepo/WaymoObjectDetection/DatasetTools/loadwaymodict.py

from glob import glob
import time
import os
from pathlib import Path
import numpy as np
import cv2
import datetime
import pickle

import numpy as np

from waymo_open_dataset import dataset_pb2
#from waymo_open_dataset.protos import metrics_pb2
from waymo_open_dataset.protos import metrics_pb2, submission_pb2


#load the results from 2DObject/dockersubmission/make_objects_file_from_latency_resultsHPC.py
def loadcreatedobjectfiles(objectfilepath):
    objects = metrics_pb2.Objects()
    with open(objectfilepath, "rb") as fd:
        objects.ParseFromString(fd.read())
    print(type(objects)) #<class 'waymo_open_dataset.protos.metrics_pb2.Objects'>
    print('Got ', len(objects.objects), 'objects')
    return objects

# def createsubmissionfromobject(objects, outputsubmissionfilepath, prefix):
#     submission = submission_pb2.Submission()
#     submission.task = submission_pb2.Submission.DETECTION_3D #DETECTION_2D
#     submission.account_name = 'kaikai.liu@sjsu.edu'
#     submission.authors.append('Kaikai Liu')
#     submission.affiliation = 'San Jose State University'
#     # 'fake' unique_method_name should not be too long
#     submission.unique_method_name = prefix
#     submission.description = 'mm3d'
#     submission.method_link = "https://drive.google.com/drive/folders/1qy8FmmrQfCKjg0G6nKVjME6tmZnlLBTz?usp=sharing"
#     submission.sensor_type = submission_pb2.Submission.LIDAR_ALL #CAMERA_ALL
#     submission.number_past_frames_exclude_current = 0
#     submission.number_future_frames_exclude_current = 0
#     submission.inference_results.CopyFrom(objects)
#     f = open(outputsubmissionfilepath, 'wb')  # output submission file
#     f.write(submission.SerializeToString())
#     f.close()

def createsubmissionfromobject(objects, outputsubmissionfilepath, prefix):
    submission = submission_pb2.Submission()
    submission.task = submission_pb2.Submission.DETECTION_3D #DETECTION_2D
    submission.account_name = 'lkk688@gmail.com'
    submission.authors.append('Kaikai Liu')
    submission.affiliation = 'San Jose State University'
    # 'fake' unique_method_name should not be too long
    submission.unique_method_name = prefix
    submission.description = 'mm3d'
    submission.docker_image_source = "us-west1-docker.pkg.dev/cmpelkk/mycontainers/mymm3d@sha256:d2853bdc620a4e3aa434b1d22bbec8dfcedb09e6378978872d64dd9d1ab20d1b"
    submission.method_link = "https://drive.google.com/drive/folders/1qy8FmmrQfCKjg0G6nKVjME6tmZnlLBTz?usp=sharing"
    submission.sensor_type = submission_pb2.Submission.LIDAR_ALL #CAMERA_ALL
    submission.number_past_frames_exclude_current = 0
    submission.number_future_frames_exclude_current = 0
    submission.inference_results.CopyFrom(objects)
    f = open(outputsubmissionfilepath, 'wb')  # output submission file
    f.write(submission.SerializeToString())
    f.close()

if __name__ == "__main__":
    #test the above functions: convert a Frame proto into a dictionary
    #convert_frame_to_dict
    # base_dir="/data/cmpe249-f20/WaymoKittitMulti/validationalldicts"
    # base_dir = Path(base_dir)
    # filename="0_step1_10203656353524179475_7625_000_7645_000.npz"

    #loadonedictfile(base_dir, filename)# load our own created dictionary file (compressed)

    nameprefix = "609mm3d3classvalall"#"609mm3dvalall" #"609mm3d3classvalall"#
    objectfilepath = "/home/010796032/MyRepo/myoutputs/"+nameprefix+"_dicvalall3dobjects"
    resultobjects=loadcreatedobjectfiles(objectfilepath)
    outputsubmissionfilepath="/home/010796032/MyRepo/myoutputs/"+nameprefix+"_dicvalall3dsubmission5.bin"
    createsubmissionfromobject(resultobjects, outputsubmissionfilepath, nameprefix)

