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

# def createsubmissionfromobject(objects, outputsubmissionfilepath, prefix):
#     submission = submission_pb2.Submission()
#     submission.task = submission_pb2.Submission.DETECTION_2D
#     submission.account_name = 'lkk688@gmail.com'
#     submission.authors.append('Kaikai Liu')
#     submission.affiliation = 'San Jose State University'
#     # 'fake' unique_method_name should not be too long
#     submission.unique_method_name = prefix
#     submission.docker_image_source = "us-west1-docker.pkg.dev/cmpelkk/mycontainers/mytftorchdetectron2@sha256:41bbb1469346d2fb5a31291b76e3d9857e5244ce83a8ae18f199ea96a15a2bcb"
#     #Detectron2: "gcr.io/cmpelkk/mytftorch@sha256:629f60dd1b93c1750f909dd3064292f51d04db3b5cc041cf6fdfb26a49028c61"
#     submission.description = 'Detectron2'
#     submission.method_link = "https://drive.google.com/drive/folders/1qy8FmmrQfCKjg0G6nKVjME6tmZnlLBTz?usp=sharing"
#     submission.sensor_type = submission_pb2.Submission.CAMERA_ALL
#     submission.number_past_frames_exclude_current = 0
#     submission.number_future_frames_exclude_current = 0
#     submission.inference_results.CopyFrom(objects)
#     f = open(outputsubmissionfilepath, 'wb')  # output submission file
#     f.write(submission.SerializeToString())
#     f.close()

def createsubmissionfromobject(objects, outputsubmissionfilepath, prefix):
    submission = submission_pb2.Submission()
    submission.task = submission_pb2.Submission.DETECTION_2D
    submission.account_name = 'lkk688@gmail.com'
    submission.authors.append('Kaikai Liu')
    submission.affiliation = 'San Jose State University'
    # 'fake' unique_method_name should not be too long
    submission.unique_method_name = prefix
    submission.docker_image_source = "us-west1-docker.pkg.dev/cmpelkk/mycontainers/mytftorchvision@sha256:354ae11bb8947bfc2295aabb789b54b98ea92100425a061084723a9f00b6a35b"
    #Detectron2: "gcr.io/cmpelkk/mytftorch@sha256:629f60dd1b93c1750f909dd3064292f51d04db3b5cc041cf6fdfb26a49028c61"
    submission.description = 'Torchvision'
    submission.method_link = "https://drive.google.com/drive/folders/1qy8FmmrQfCKjg0G6nKVjME6tmZnlLBTz?usp=sharing"
    submission.sensor_type = submission_pb2.Submission.CAMERA_ALL
    submission.number_past_frames_exclude_current = 0
    submission.number_future_frames_exclude_current = 0
    submission.inference_results.CopyFrom(objects)
    f = open(outputsubmissionfilepath, 'wb')  # output submission file
    f.write(submission.SerializeToString())
    f.close()

# def createsubmissionfromobject(objects, outputsubmissionfilepath, prefix):
#     submission = submission_pb2.Submission()
#     submission.task = submission_pb2.Submission.DETECTION_2D
#     submission.account_name = 'lkk688@gmail.com'
#     submission.authors.append('Kaikai Liu')
#     submission.affiliation = 'San Jose State University'
#     # 'fake' unique_method_name should not be too long
#     submission.unique_method_name = prefix
#     submission.docker_image_source = "us-west1-docker.pkg.dev/cmpelkk/mycontainers/mytf2@sha256:a9a6ac9f9fc71ad6c872e21e527ff8421c62e3d4fe800c9538213b108b698764"
#     #Detectron2: "gcr.io/cmpelkk/mytftorch@sha256:629f60dd1b93c1750f909dd3064292f51d04db3b5cc041cf6fdfb26a49028c61"
#     submission.description = 'Tensorflow2 od'
#     submission.method_link = "https://drive.google.com/drive/folders/1qy8FmmrQfCKjg0G6nKVjME6tmZnlLBTz?usp=sharing"
#     submission.sensor_type = submission_pb2.Submission.CAMERA_ALL
#     submission.number_past_frames_exclude_current = 0
#     submission.number_future_frames_exclude_current = 0
#     submission.inference_results.CopyFrom(objects)
#     f = open(outputsubmissionfilepath, 'wb')  # output submission file
#     f.write(submission.SerializeToString())
#     f.close()

if __name__ == "__main__":
    #test the above functions: convert a Frame proto into a dictionary
    #convert_frame_to_dict
    # base_dir="/data/cmpe249-f20/WaymoKittitMulti/validationalldicts"
    # base_dir = Path(base_dir)
    # filename="0_step1_10203656353524179475_7625_000_7645_000.npz"

    #loadonedictfile(base_dir, filename)# load our own created dictionary file (compressed)

    nameprefix = "611tfvalall"#"610torchvisiontestall" #"610dtrn2testall"#"611tftestall" #"609torchvisionvalall"# #"610torchvisiontestall"#"609torchvisionvalall" #"0603dtrn2valall" #"610torchvisiontestall" #"0603dtrn2valall"#"609torchvisionvalall"#"609dtrn2valall"#"609mmdet35valall"#"531mmdet27valall"#"0603dtrn2valall"
    objectfilepath = "/home/010796032/MyRepo/myoutputs/"+nameprefix+"_diccameraobjects" #"_diccameraobjects" #0603dtrn2valall_dicvalallcameraobjects
    resultobjects=loadcreatedobjectfiles(objectfilepath)
    outputsubmissionfilepath="/home/010796032/MyRepo/myoutputs/"+nameprefix+"_diccamerasubmission5.bin"
    createsubmissionfromobject(resultobjects, outputsubmissionfilepath, nameprefix)

