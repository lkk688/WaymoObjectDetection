from waymo_open_dataset.protos import metrics_pb2, submission_pb2
import tensorflow.compat.v1 as tf
from waymo_open_dataset import label_pb2
from waymo_open_dataset import dataset_pb2
from waymo_open_dataset import dataset_pb2 as open_dataset
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
import pickle

category_index = {1: {'id': 1, 'name': 'VEHICLE'},
                  2: {'id': 2, 'name': 'PEDESTRIAN'},
                  3: {'id': 3, 'name': 'SIGN'},
                  4: {'id': 4, 'name': 'CYCLIST'}}
Threshold = 0.2

#ref from https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/utils/frame_utils.py
def convert_frame_to_dict_cameras(frame):
    """Convert the frame proto into a dict of numpy arrays.
    The keys, shapes, and data types are:
      POSE: 4x4 float32 array
      TIMESTAMP: int64 scalar
      For each camera:
        <CAMERA_NAME>_IMAGE: HxWx3 uint8 array
        <CAMERA_NAME>_INTRINSIC: 9 float32 array
        <CAMERA_NAME>_EXTRINSIC: 4x4 float32 array
        <CAMERA_NAME>_WIDTH: int64 scalar
        <CAMERA_NAME>_HEIGHT: int64 scalar
        <CAMERA_NAME>_SDC_VELOCITY: 6 float32 array
        <CAMERA_NAME>_POSE: 4x4 float32 array
        <CAMERA_NAME>_POSE_TIMESTAMP: float32 scalar
        <CAMERA_NAME>_ROLLING_SHUTTER_DURATION: float32 scalar
        <CAMERA_NAME>_ROLLING_SHUTTER_DIRECTION: int64 scalar
        <CAMERA_NAME>_CAMERA_TRIGGER_TIME: float32 scalar
        <CAMERA_NAME>_CAMERA_READOUT_DONE_TIME: float32 scalar
    NOTE: This function only works in eager mode for now.
    See the LaserName.Name and CameraName.Name enums in dataset.proto for the
    valid lidar and camera name strings that will be present in the returned
    dictionaries.
    Args:
      frame: open dataset frame
    Returns:
      Dict from string field name to numpy ndarray.
    """
    data_dict = {}

    # Save the H x W x 3 RGB image for each camera, extracted from JPEG.
    for im in frame.images:
        cam_name_str = dataset_pb2.CameraName.Name.Name(im.name)
        data_dict[f'{cam_name_str}_IMAGE'] = tf.io.decode_jpeg(
            im.image).numpy()
        # data_dict[f'{cam_name_str}_SDC_VELOCITY'] = np.array([
        #     im.velocity.v_x, im.velocity.v_y, im.velocity.v_z, im.velocity.w_x,
        #     im.velocity.w_y, im.velocity.w_z
        # ], np.float32)
        # data_dict[f'{cam_name_str}_POSE'] = np.reshape(
        #     np.array(im.pose.transform, np.float32), (4, 4))
        # data_dict[f'{cam_name_str}_POSE_TIMESTAMP'] = np.array(
        #     im.pose_timestamp, np.float32)
        # data_dict[f'{cam_name_str}_ROLLING_SHUTTER_DURATION'] = np.array(
        #     im.shutter)
        # data_dict[f'{cam_name_str}_CAMERA_TRIGGER_TIME'] = np.array(
        #     im.camera_trigger_time)
        # data_dict[f'{cam_name_str}_CAMERA_READOUT_DONE_TIME'] = np.array(
        #     im.camera_readout_done_time)

    # Save the intrinsics, 4x4 extrinsic matrix, width, and height of each camera.
    # for c in frame.context.camera_calibrations:
    #     cam_name_str = dataset_pb2.CameraName.Name.Name(c.name)
    #     print(f'Camera name: {cam_name_str}, width: {c.width}, height: {c.height}')
        # data_dict[f'{cam_name_str}_INTRINSIC'] = np.array(
        #     c.intrinsic, np.float32)
        # data_dict[f'{cam_name_str}_EXTRINSIC'] = np.reshape(
        #     np.array(c.extrinsic.transform, np.float32), [4, 4])
        # data_dict[f'{cam_name_str}_WIDTH'] = np.array(c.width)
        # data_dict[f'{cam_name_str}_HEIGHT'] = np.array(c.height)
        # data_dict[f'{cam_name_str}_ROLLING_SHUTTER_DIRECTION'] = np.array(
        #     c.rolling_shutter_direction)

    # data_dict['POSE'] = np.reshape(
    #     np.array(frame.pose.transform, np.float32), (4, 4))
    data_dict['TIMESTAMP'] = np.array(frame.timestamp_micros)

    return data_dict


def evaluateallframescreatesubmission(frames, outputsubmissionfilepath, outputfile="./output_video1.mp4"):
    array_len = len(frames) #4931 frames for validation_0000
    # 20, 200 frames in one file, downsample by 10
    print("Frames lenth:", array_len)
    print("Final_array type:", type(frames))  # class 'list'

    objects = metrics_pb2.Objects()  # submission objects

    frame_width = 1920
    frame_height = 1280
    out = cv2.VideoWriter(outputfile, cv2.VideoWriter_fourcc(
        'M', 'P', '4', 'V'), 5, (frame_width, frame_height))
    fps = FPS().start()

    wod_latency_submission.initialize_model()

    required_field = wod_latency_submission.DATA_FIELDS
    print(required_field)

    for frameid in range(array_len):
        #frameid = 5
        print("frameid:", frameid)
        # {'key':key, 'context_name':context_name, 'framedict':framedict}
        currentframe=frames[frameid]
        convertedframesdict = convert_frame_to_dict_cameras(currentframe) #data_array[frameid]
        frame_timestamp_micros = convertedframesdict['TIMESTAMP']#['key']
        context_name = currentframe.context.name #convertedframesdict['context_name']
        #framedict = convertedframesdict['framedict']
        # 10017090168044687777_6380_000_6400_000
        #print('context_name:', context_name)
        # print('frame_timestamp_micros:', frame_timestamp_micros)  # 1550083467346370

        #result = wod_latency_submission.run_model(framedict[required_field[0]], framedict[required_field[1]])
        #result = wod_latency_submission.run_model(**framedict)
        #Front_image = framedict[required_field[0]]
        start_time = time.time()
        #result = wod_latency_submission.run_model(Front_image)
        result = wod_latency_submission.run_model(**convertedframesdict)#All images
        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Inference time: ' + str(elapsed_time) + 's')
        # print(result)

        createsubmisionobject(objects, result['boxes'], result['classes'],
                              result['scores'], context_name, frame_timestamp_micros)

        # Save the original image
        #output_path = "./test.png"
        #visualization_util.save_image_array_as_png(Front_image, output_path)

        Front_image = convertedframesdict[required_field[0]]
        image_np_with_detections = Front_image.copy()
        visualization_util.visualize_boxes_and_labels_on_image_array(image_np_with_detections, result['boxes'], result['classes'], result['scores'], category_index, use_normalized_coordinates=False,
                                                                     max_boxes_to_draw=200,
                                                                     min_score_thresh=Threshold,
                                                                     agnostic_mode=False)
        display_str=f'Inference time: {str(elapsed_time*1000)}ms, context_name: {context_name}, timestamp_micros: {frame_timestamp_micros}'
        visualization_util.draw_text_on_image(image_np_with_detections, 0, 0, display_str, color='black')
        #visualization_util.save_image_array_as_png(image_np_with_detections, outputfile)

        name = './Test_data/frame' + str(frameid) + '.jpg'
        #print ('Creating\...' + name)
        # cv2.imwrite(name, image_np_with_detections) #write to image folder
        fps.update()
        #out.write(image_np_with_detections)
        out.write(cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR))
        #cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    out.release()

    with open('objectsresult.pickle', 'wb') as f:
        pickle.dump(objects, f)

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
    f = open(outputsubmissionfilepath, 'wb')  # output submission file
    f.write(submission.SerializeToString())
    f.close()

    now = datetime.datetime.now()
    print("Finished validation, current date and time : ")
    print(now.strftime("%Y-%m-%d %H:%M:%S"))


INSTANCE_pb2 = {
    'Unknown': label_pb2.Label.TYPE_UNKNOWN, 'Vehicles': label_pb2.Label.TYPE_VEHICLE, 'Pedestrians': label_pb2.Label.TYPE_PEDESTRIAN, 'Cyclists': label_pb2.Label.TYPE_CYCLIST
}
INSTANCEindex_pb2 = {
    0: label_pb2.Label.TYPE_UNKNOWN, 1: label_pb2.Label.TYPE_VEHICLE, 2: label_pb2.Label.TYPE_PEDESTRIAN,  3: label_pb2.Label.TYPE_SIGN, 4: label_pb2.Label.TYPE_CYCLIST
}

INSTANCE_CATEGORY_NAMES = [
    'Unknown', 'Vehicles', 'Pedestrians', 'Cyclists'
]


def createsubmisionobject(objects, boxes, pred_cls, scores, context_name, frame_timestamp_micros):
    total_boxes = len(boxes)
    for i in range(total_boxes):  # patch in pred_bbox:
        label = pred_cls[i]
        #(center_x, center_y, width, height) in image size
        bbox = boxes[i] #[1246.5217, 750.64905, 113.49747, 103.9653]
        score = scores[i]
        o = metrics_pb2.Object()
        o.context_name = context_name  # frame.context.name
        # frame.timestamp_micros)
        o.frame_timestamp_micros = int(frame_timestamp_micros)
        o.camera_name = dataset_pb2.CameraName.FRONT
        o.score = score

        # Populating box and score.
        box = label_pb2.Label.Box()
        box.center_x = bbox[0]
        box.center_y = bbox[1]
        box.width=bbox[2]
        box.length=bbox[3]
        # box.length = bbox[1][0] - bbox[0][0]
        # box.width = bbox[1][1] - bbox[0][1]
        # box.center_x = bbox[0][0] + box.length * 0.5
        # box.center_y = bbox[0][1] + box.width * 0.5

        o.object.box.CopyFrom(box)
        o.object.detection_difficulty_level = label_pb2.Label.LEVEL_1
        o.object.num_lidar_points_in_box = 100
        # INSTANCE_CATEGORY_NAMES.index(label) #INSTANCE_pb2[label]
        o.object.type = INSTANCEindex_pb2[label] #INSTANCE_pb2[label]
        # print(
        #     f'Object type label: {label}, {INSTANCE_pb2[label]}, {INSTANCE_CATEGORY_NAMES.index(label)}')
        assert o.object.type != label_pb2.Label.TYPE_UNKNOWN
        objects.objects.append(o)
        # return o

def evaluateallframesgtfakesubmission(frames, outputsubmissionfilepath, outputfile="./output_video1.mp4"):
    array_len = len(frames) #4931 frames for validation_0000
    # 20, 200 frames in one file, downsample by 10
    print("Frames lenth:", array_len)
    print("Final_array type:", type(frames))  # class 'list'

    objects = metrics_pb2.Objects()  # submission objects

    frame_width = 1920
    frame_height = 1280
    out = cv2.VideoWriter(outputfile, cv2.VideoWriter_fourcc(
        'M', 'P', '4', 'V'), 5, (frame_width, frame_height))
    fps = FPS().start()

    #wod_latency_submission.initialize_model()

    #required_field = wod_latency_submission.DATA_FIELDS
    #print(required_field)
    #Allconvertedframesdict=[]

    for frameid in range(array_len):
        #frameid = 5
        print("frameid:", frameid)
        # {'key':key, 'context_name':context_name, 'framedict':framedict}
        frame=frames[frameid]
        convertedframesdict = convert_frame_to_dict_cameras(frame) #data_array[frameid]
        #Allconvertedframesdict.append(convertedframesdict)
        frame_timestamp_micros = convertedframesdict['TIMESTAMP']#['key']
        context_name = frame.context.name

        o_list = []
        boundingbox=[]
        boxscore=[]
        boxtype=[]
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
                boundingbox.append([box.center_x, box.center_y, box.length, box.width]) #width height
                o.object.box.CopyFrom(box)
                # This must be within [0.0, 1.0]. It is better to filter those boxes with
                # small scores to speed up metrics computation.
                o.score = 0.9
                boxscore.append(o.score)
                # Use correct type.
                o.object.type = gt_label.type
                boxtype.append(o.object.type)
                o_list.append(o)
                print(f'Camera labelname: {camera_labels.name}, object type: { gt_label.type}, box:{box}')

        # Save the original image
        #output_path = "./test.png"
        #visualization_util.save_image_array_as_png(Front_image, output_path)
        boundingbox=np.array(boundingbox)
        boxscore=np.array(boxscore)
        boxtype=np.array(boxtype).astype(np.uint8)

        Front_image = convertedframesdict['FRONT_IMAGE']
        image_np_with_detections = Front_image.copy()
        visualization_util.visualize_boxes_and_labels_on_image_array(image_np_with_detections, boundingbox, boxtype, boxscore, category_index, use_normalized_coordinates=False,
                                                                     max_boxes_to_draw=200,
                                                                     min_score_thresh=Threshold,
                                                                     agnostic_mode=False)
        display_str=f'context_name: {context_name}, timestamp_micros: {frame_timestamp_micros}'
        visualization_util.draw_text_on_image(image_np_with_detections, 0, 0, display_str, color='black')
        #visualization_util.save_image_array_as_png(image_np_with_detections, outputfile)

        name = './Test_data/frame' + str(frameid) + '.jpg'
        #print ('Creating\...' + name)
        # cv2.imwrite(name, image_np_with_detections) #write to image folder
        fps.update()
        #out.write(image_np_with_detections)
        out.write(cv2.cvtColor(image_np_with_detections, cv2.COLOR_RGB2BGR))
        #cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    out.release()

    with open('objectsresult_gtvalall.pickle', 'wb') as f:
        pickle.dump(objects, f)
    # with open('allframedics.pickle', 'wb') as f:
    #     pickle.dump(Allconvertedframesdict, f)

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
    f = open(outputsubmissionfilepath, 'wb')  # output submission file
    f.write(submission.SerializeToString())
    f.close()

    now = datetime.datetime.now()
    print("Finished validation, current date and time : ")
    print(now.strftime("%Y-%m-%d %H:%M:%S"))

def loadWaymoValidationFrames(PATH):
    #validation_folders = ["validation_0000"]#,"validation_0005"]
    # ["validation_0007","validation_0006","validation_0005","validation_0004","validation_0003","validation_0002","validation_0001","validation_0000"]
    validation_folders = ["validation_0000", "validation_0001", "validation_0002","validation_0003", "validation_0004", "validation_0005", "validation_0006", "validation_0007"]
    data_files = [path for x in validation_folders for path in glob(
        os.path.join(PATH, x, "*.tfrecord"))]
    print(data_files)  # all TFRecord file list
    print(len(data_files))
    # create a list of dataset for each TFRecord file
    dataset = [tf.data.TFRecordDataset(
        FILENAME, compression_type='') for FILENAME in data_files]
    # store all frames = total number of TFrecord files * 40 frame(each TFrecord)
    frames = []
    for i, data_file in enumerate(dataset):
        print("Datafile: ", i)  # Each TFrecord file
        # Create frame based on Waymo API, 199 frames per TFrecord (20s, 10Hz)
        for idx, data in enumerate(data_file):
            #             if idx % 5 != 0: #Downsample every 5 images, reduce to 2Hz, total around 40 frames
            #                 continue
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            frames.append(frame)
    return frames

def loadobjectresultspkl():
    filename = "/Developer/MyRepo/WaymoObjectDetection/objectsresult.pickle"
    infile = open(filename,'rb')
    new_dict = pickle.load(infile)
    infile.close()

    print(type(new_dict))#waymo_open_dataset.protos.metrics_pb2.Objects
    

if __name__ == "__main__":
    print("Loading Waymo validation frames...")
    
    PATH = "/DATA5T/Dataset/WaymoDataset/"
    waymovalidationframes = loadWaymoValidationFrames(PATH)
    #mywaymovaldataset = myNewWaymoDataset(PATH, waymovalidationframes, get_transform(train=False))
    print("Total validation frames: ", len(waymovalidationframes))

    outputfile = "./output/output_video_tf130kval0.mp4"
    outputsubmissionfilepath = './output/tf130k_val0.bin'

    outputfile = "/Developer/MyRepo/WaymoObjectDetection/output/output_video_mmhpc25valall.mp4"
    outputsubmissionfilepath = '/Developer/MyRepo/WaymoObjectDetection/output/mmhpc25_valall.bin'
    #evaluateallframescreatesubmission(waymovalidationframes,outputsubmissionfilepath, outputfile)

    outputfile = "/Developer/MyRepo/WaymoObjectDetection/output/output_video_gtnewvalall.mp4"
    outputsubmissionfilepath = '/Developer/MyRepo/WaymoObjectDetection/output/gt_valall.bin'
    evaluateallframesgtfakesubmission(waymovalidationframes,outputsubmissionfilepath, outputfile)