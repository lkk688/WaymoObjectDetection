import wod_latency_submission

from glob import glob
import time
import os
from pathlib import Path
import numpy as np

import visualization_util

from object_detection.utils import label_map_util
from imutils.video import FPS
import imutils
import cv2
import datetime

category_index = { 1: {'id': 1, 'name': 'VEHICLE'}, \
                        2: {'id': 2, 'name': 'PEDESTRIAN'}, \
                        3: {'id': 3, 'name': 'SIGN'}, \
                        4: {'id': 4, 'name': 'CYCLIST'}}
Threshold=0.2

def evaluatesingleframe(base_dir, filename, frameid, outputfile="./testresult.png"):
    Final_array = np.load(base_dir / filename,
                          allow_pickle=True, mmap_mode='r')
    data_array = Final_array['arr_0']
    array_len = len(data_array)
    # 20, 200 frames in one file, downsample by 10
    print("Final_array lenth:", array_len)
    print("Final_array type:", type(data_array))  # numpy.ndarray

    # for frameid in range(array_len):
    #frameid = 5
    print("frameid:", frameid)
    # {'key':key, 'context_name':context_name, 'framedict':framedict}
    convertedframesdict = data_array[frameid]
    frame_timestamp_micros = convertedframesdict['key']
    context_name = convertedframesdict['context_name']
    framedict = convertedframesdict['framedict']
    # 10017090168044687777_6380_000_6400_000
    print('context_name:', context_name)
    print('frame_timestamp_micros:', frame_timestamp_micros)  # 1550083467346370

    start_time = time.time()
    wod_latency_submission.initialize_model()
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Model Inalization elapsed time: ' + str(elapsed_time) + 's')

    required_field = wod_latency_submission.DATA_FIELDS
    print(required_field)

    #result = wod_latency_submission.run_model(framedict[required_field[0]], framedict[required_field[1]])
    #result = wod_latency_submission.run_model(**framedict)
    Front_image = framedict[required_field[0]]
    start_time = time.time()
    result = wod_latency_submission.run_model(Front_image)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Inference time: ' + str(elapsed_time) + 's')
    #print(result)

    #Save the original image
    #output_path = "./test.png"
    #visualization_util.save_image_array_as_png(Front_image, output_path)

    image_np_with_detections = Front_image.copy()
    # category_index = { 1: {'id': 1, 'name': 'VEHICLE'}, \
    #                         2: {'id': 2, 'name': 'PEDESTRIAN'}, \
    #                         3: {'id': 3, 'name': 'SIGN'}, \
    #                         4: {'id': 4, 'name': 'CYCLIST'}}
    # label_map_path = '2DObject/tfobjectdetection/waymo_labelmap.txt'
    # label_map = label_map_util.load_labelmap(label_map_path)
    # categories = label_map_util.convert_label_map_to_categories(
    #     label_map,
    #     max_num_classes=label_map_util.get_max_label_map_index(label_map),
    #     use_display_name=True)
    # category_index = label_map_util.create_category_index(categories)
    visualization_util.visualize_boxes_and_labels_on_image_array(image_np_with_detections, result['boxes'], result['classes'], result['scores'], category_index, use_normalized_coordinates=False,
                                                             max_boxes_to_draw=200,
                                                             min_score_thresh=Threshold,
                                                             agnostic_mode=False)
    visualization_util.save_image_array_as_png(
        image_np_with_detections, outputfile)
# plt.figure(figsize=(12,16))
# plt.imshow(image_np_with_detections)
# plt.show()


def evaluateallframes(base_dir, filename, outputfile="./output_video1.mp4"):
    Final_array = np.load(base_dir / filename,
                          allow_pickle=True, mmap_mode='r')
    data_array = Final_array['arr_0']
    array_len = len(data_array)
    # 20, 200 frames in one file, downsample by 10
    print("Final_array lenth:", array_len)
    print("Final_array type:", type(data_array))  # numpy.ndarray

    frame_width=1920
    frame_height=1280
    out = cv2.VideoWriter(outputfile,cv2.VideoWriter_fourcc('M','P','4','V'), 2, (frame_width,frame_height))
    fps = FPS().start()

    wod_latency_submission.initialize_model()

    required_field = wod_latency_submission.DATA_FIELDS
    print(required_field)

    for frameid in range(array_len):
        #frameid = 5
        print("frameid:", frameid)
        # {'key':key, 'context_name':context_name, 'framedict':framedict}
        convertedframesdict = data_array[frameid]
        frame_timestamp_micros = convertedframesdict['key']
        context_name = convertedframesdict['context_name']
        framedict = convertedframesdict['framedict']
        # 10017090168044687777_6380_000_6400_000
        #print('context_name:', context_name)
        #print('frame_timestamp_micros:', frame_timestamp_micros)  # 1550083467346370

        #result = wod_latency_submission.run_model(framedict[required_field[0]], framedict[required_field[1]])
        #result = wod_latency_submission.run_model(**framedict)
        Front_image = framedict[required_field[0]]
        start_time = time.time()
        result = wod_latency_submission.run_model(Front_image)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Inference time: ' + str(elapsed_time) + 's')
        #print(result)

        #Save the original image
        #output_path = "./test.png"
        #visualization_util.save_image_array_as_png(Front_image, output_path)

        image_np_with_detections = Front_image.copy()
        visualization_util.visualize_boxes_and_labels_on_image_array(image_np_with_detections, result['boxes'], result['classes'], result['scores'], category_index, use_normalized_coordinates=False,
                                                                max_boxes_to_draw=200,
                                                                min_score_thresh=Threshold,
                                                                agnostic_mode=False)
        #visualization_util.save_image_array_as_png(image_np_with_detections, outputfile)

        name = './Test_data/frame' + str(frameid) + '.jpg'
        #print ('Creating\...' + name) 
        #cv2.imwrite(name, image_np_with_detections) #write to image folder
        fps.update()
        out.write(image_np_with_detections)

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    out.release()

def evaluateallframescreatesubmission(base_dir, filename, outputsubmissionfilepath, outputfile="./output_video1.mp4"):
    Final_array = np.load(base_dir / filename,
                          allow_pickle=True, mmap_mode='r')
    data_array = Final_array['arr_0']
    array_len = len(data_array)
    # 20, 200 frames in one file, downsample by 10
    print("Final_array lenth:", array_len)
    print("Final_array type:", type(data_array))  # numpy.ndarray

    objects = metrics_pb2.Objects() #submission objects

    frame_width=1920
    frame_height=1280
    out = cv2.VideoWriter(outputfile,cv2.VideoWriter_fourcc('M','P','4','V'), 2, (frame_width,frame_height))
    fps = FPS().start()

    wod_latency_submission.initialize_model()

    required_field = wod_latency_submission.DATA_FIELDS
    print(required_field)

    for frameid in range(array_len):
        #frameid = 5
        print("frameid:", frameid)
        # {'key':key, 'context_name':context_name, 'framedict':framedict}
        convertedframesdict = data_array[frameid]
        frame_timestamp_micros = convertedframesdict['key']
        context_name = convertedframesdict['context_name']
        framedict = convertedframesdict['framedict']
        # 10017090168044687777_6380_000_6400_000
        #print('context_name:', context_name)
        #print('frame_timestamp_micros:', frame_timestamp_micros)  # 1550083467346370

        #result = wod_latency_submission.run_model(framedict[required_field[0]], framedict[required_field[1]])
        #result = wod_latency_submission.run_model(**framedict)
        Front_image = framedict[required_field[0]]
        start_time = time.time()
        result = wod_latency_submission.run_model(Front_image)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Inference time: ' + str(elapsed_time) + 's')
        #print(result)

        createsubmisionobject(objects, result['boxes'],result['classes'], result['scores'], context_name, frame_timestamp_micros)

        #Save the original image
        #output_path = "./test.png"
        #visualization_util.save_image_array_as_png(Front_image, output_path)

        image_np_with_detections = Front_image.copy()
        visualization_util.visualize_boxes_and_labels_on_image_array(image_np_with_detections, result['boxes'], result['classes'], result['scores'], category_index, use_normalized_coordinates=False,
                                                                max_boxes_to_draw=200,
                                                                min_score_thresh=Threshold,
                                                                agnostic_mode=False)
        #visualization_util.save_image_array_as_png(image_np_with_detections, outputfile)

        name = './Test_data/frame' + str(frameid) + '.jpg'
        #print ('Creating\...' + name) 
        #cv2.imwrite(name, image_np_with_detections) #write to image folder
        fps.update()
        out.write(image_np_with_detections)

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    out.release()

    submission = submission_pb2.Submission()
    submission.task = submission_pb2.Submission.DETECTION_2D 
    submission.account_name = 'kaikai.liu@sjsu.edu'
    submission.authors.append('Kaikai Liu')
    submission.affiliation = 'None'
    submission.unique_method_name = 'torchvisionfaster'
    submission.description = 'none'
    submission.method_link = "empty method"
    submission.sensor_type = submission_pb2.Submission.CAMERA_ALL
    submission.number_past_frames_exclude_current = 0
    submission.number_future_frames_exclude_current = 0
    submission.inference_results.CopyFrom(objects)
    f = open(outputsubmissionfilepath, 'wb') #output submission file
    f.write(submission.SerializeToString())
    f.close()
    
    now = datetime.datetime.now()
    print ("Finished validation, current date and time : ")
    print (now.strftime("%Y-%m-%d %H:%M:%S"))

from waymo_open_dataset.protos import metrics_pb2,submission_pb2
from waymo_open_dataset import dataset_pb2 
from waymo_open_dataset import label_pb2  
INSTANCE_pb2 = {
    'Unknown':label_pb2.Label.TYPE_UNKNOWN, 'Vehicles':label_pb2.Label.TYPE_VEHICLE, 'Pedestrians':label_pb2.Label.TYPE_PEDESTRIAN, 'Cyclists':label_pb2.Label.TYPE_CYCLIST
}
INSTANCE_CATEGORY_NAMES = [
    'Unknown', 'Vehicles', 'Pedestrians', 'Cyclists'
]
def createsubmisionobject(objects, boxes,pred_cls, scores, context_name, frame_timestamp_micros):
    total_boxes=len(boxes)
    for i in range(total_boxes):#patch in pred_bbox:
        label=pred_cls[i]
        bbox=boxes[i]
        score = scores[i]
        o = metrics_pb2.Object()
        o.context_name = context_name#frame.context.name
        o.frame_timestamp_micros = int(frame_timestamp_micros)#frame.timestamp_micros)
        o.camera_name = dataset_pb2.CameraName.FRONT
        o.score = score
        
        # Populating box and score.
        box = label_pb2.Label.Box()
        box.length = bbox[1][0] - bbox[0][0]
        box.width = bbox[1][1] - bbox[0][1]
        box.center_x = bbox[0][0] + box.length * 0.5
        box.center_y = bbox[0][1] + box.width * 0.5

        o.object.box.CopyFrom(box)
        o.object.detection_difficulty_level = label_pb2.Label.LEVEL_1
        o.object.num_lidar_points_in_box = 100
        o.object.type = INSTANCE_pb2[label]# INSTANCE_CATEGORY_NAMES.index(label) #INSTANCE_pb2[label]
        print(f'Object type label: {label}, {INSTANCE_pb2[label]}, {INSTANCE_CATEGORY_NAMES.index(label)}')
        assert o.object.type != label_pb2.Label.TYPE_UNKNOWN
        objects.objects.append(o)
        #return o

if __name__ == "__main__":
    # test the above functions: convert a Frame proto into a dictionary
    # convert_frame_to_dict
    # "/data/cmpe249-f20/WaymoKittitMulti/dict_train0"

    #Evaluate a single frame
    base_dir = "/Developer/3DObject"
    base_dir = Path(base_dir)
    filename = "1_step10_10017090168044687777_6380_000_6400_000.npz"
    frameid=10
    #evaluatesingleframe(base_dir, filename, frameid)

    outputfile="./output_video1.mp4"
    evaluateallframes(base_dir, filename,outputfile)



