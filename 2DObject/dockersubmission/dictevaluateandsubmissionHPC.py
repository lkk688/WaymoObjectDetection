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

def loadonedictfile(base_dir, filename):
    Final_array=np.load(base_dir / filename, allow_pickle=True, mmap_mode='r')
    data_array=Final_array['arr_0']
    array_len=len(data_array)
    print("Final_array lenth:", array_len)
    print("Final_array type:", type(data_array))

    #for frameid in range(array_len):
    frameid=0
    print("frameid:", frameid)
    convertedframesdict = data_array[frameid] #{'key':key, 'context_name':context_name, 'framedict':framedict}
    frame_timestamp_micros=convertedframesdict['key']
    context_name=convertedframesdict['context_name']
    framedict=convertedframesdict['framedict']
    print('context_name:', context_name)
    print('frame_timestamp_micros:', frame_timestamp_micros)
    for key, value in framedict.items():
        print(key)
    print(type(framedict['FRONT_IMAGE']))
    print(framedict['FRONT_IMAGE'].shape)

#load the results from 2DObject/dockersubmission/make_objects_file_from_latency_resultsHPC.py
def loadcreatedobjectfiles(objectfilepath):
    objects = metrics_pb2.Objects()
    with open(objectfilepath, "rb") as fd:
        objects.ParseFromString(fd.read())
    print(type(objects)) #<class 'waymo_open_dataset.protos.metrics_pb2.Objects'>
    print('Got ', len(objects.objects), 'objects')
    return objects

def createsubmissionfromobject(objects, outputsubmissionfilepath, prefix):
    submission = submission_pb2.Submission()
    submission.task = submission_pb2.Submission.DETECTION_2D
    submission.account_name = 'kaikai.liu@sjsu.edu'
    submission.authors.append('Kaikai Liu')
    submission.affiliation = 'San Jose State University'
    # 'fake' unique_method_name should not be too long
    submission.unique_method_name = prefix
    submission.description = 'none'
    submission.method_link = "empty method"
    submission.sensor_type = submission_pb2.Submission.CAMERA_ALL
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

    nameprefix = "609dtrn2valall"#"609mmdet35valall"#"531mmdet27valall"#"0603dtrn2valall"
    objectfilepath = "/home/010796032/MyRepo/myoutputs/"+nameprefix+"_dicvalallcameraobjects"
    resultobjects=loadcreatedobjectfiles(objectfilepath)
    outputsubmissionfilepath="/home/010796032/MyRepo/myoutputs/"+nameprefix+"_dicvalallcamerasubmission.bin"
    createsubmissionfromobject(resultobjects, outputsubmissionfilepath, nameprefix)

