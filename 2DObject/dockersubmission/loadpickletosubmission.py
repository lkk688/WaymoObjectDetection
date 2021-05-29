import datetime
import pickle
import numpy as np
from waymo_open_dataset.protos import metrics_pb2, submission_pb2

filename = "/home/010796032/MyRepo/myoutputs/outputallcamera0527_detectron899k_valall.pickle"
infile = open(filename,'rb')
objects = pickle.load(infile)

print(len(objects.objects))
print(type(objects.objects))#<class 'google.protobuf.pyext._message.RepeatedCompositeContainer'>

outputsubmissionfilepath="/home/010796032/MyRepo/myoutputs/outputallcamera0527_detectron899k_valall2.bin"

submission = submission_pb2.Submission()
submission.task = submission_pb2.Submission.DETECTION_2D
submission.account_name = 'kaikai.liu@sjsu.edu'
submission.authors.append('Kaikai Liu')
submission.affiliation = 'San Jose State University'
submission.unique_method_name = 'detectron899k' #'fake'
submission.description = 'none'
submission.method_link = "empty method"
submission.sensor_type = submission_pb2.Submission.CAMERA_ALL
submission.number_past_frames_exclude_current = 0
submission.number_future_frames_exclude_current = 0
submission.inference_results.CopyFrom(objects)
f = open(outputsubmissionfilepath, 'wb')  # output submission file
f.write(submission.SerializeToString())
f.close()