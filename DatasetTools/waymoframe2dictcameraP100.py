# ref: https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/utils/frame_utils.py

#from __future__ import absolute_import
from pathlib import Path
import os
import time
from glob import glob
# from __future__ import division
# from __future__ import print_function

import numpy as np
import tensorflow as tf

from waymo_open_dataset import dataset_pb2
from waymo_open_dataset.utils import range_image_utils
from waymo_open_dataset.utils import transform_utils


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
    #data_dict['TIMESTAMP'] = np.array(frame.timestamp_micros)

    return data_dict

#from waymo_open_dataset import dataset_pb2 as open_dataset

def extract_onesegment_todicts(fileidx, tfrecord_pathnames, step, save_folder):
    out_dir=Path(save_folder)

    segment_path = tfrecord_pathnames[fileidx]
    c_start = time.time()
    print(
        f'extracting {fileidx}, path: {segment_path}, currenttime: {c_start}')

    dataset = tf.data.TFRecordDataset(str(segment_path), compression_type='')
    #framesdict = {}  # []
    for i, data in enumerate(dataset):
        if i % step != 0:  # Downsample
            continue

        # print('.', end='', flush=True) #progress bar
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        # get one frame
        # A unique name that identifies the frame sequence
        context_name = frame.context.name
        #print('context_name:', context_name)#14824622621331930560_2395_420_2415_420, same to the tfrecord file name
        frame_timestamp_micros = str(frame.timestamp_micros)
        #print(frame_timestamp_micros)
        #context_name as the first level sub folder, frame_timestamp_micros as the second level sub folder
        #one tf record file only has one context_name
        relative_filepath = '/'.join([context_name, frame_timestamp_micros])
        filepath = out_dir / relative_filepath
        filepath.mkdir(parents=True, exist_ok=True)
        #each file os.path.join(input_dir, f'{field}.npy')
        cameradict=convert_frame_to_dict_cameras(frame)
        for key, npdata in cameradict.items():#5 cameras
            filename=key+'.npy'
            completepath=filepath / filename
            np.save(completepath, npdata)
            print(f'Saved {completepath}')
        del frame
        del data
        del cameradict
    del dataset


# def saveonedictfile(data_files, fileidx, step, out_dir):
#   framesdict = extract_onesegment_toframe(fileidx, data_files, step)
#   num_frames = len(framesdict)
#   print(num_frames)

#   Final_array=[]
#   for key, frame in framesdict.items():
#     print(key)
#     context_name = frame.context.name
#     print('context_name:', context_name)
#     #framedict=convert_frame_to_dict(frame)
#     framedict=convert_frame_to_dict_cameras(frame)
#     #print(framedict)
#     # print(type(framedict['TOP_RANGE_IMAGE_FIRST_RETURN']))#FRONT_IMAGE: <class 'numpy.ndarray'>, (1280, 1920, 3)
#     print(framedict['TOP_RANGE_IMAGE_FIRST_RETURN'].shape) #<class 'numpy.ndarray'> (64, 2650, 6)
#     print(framedict['FRONT_IMAGE'].shape)
#     convertedframesdict = {'key':key, 'context_name':context_name, 'framedict':framedict}
#     Final_array.append(convertedframesdict)
  
#   second_time = time.time()
#   print(f"Finished conversion, Execution time: { second_time - c_start }")  #Execution time: 555.8904404640198
#   filename=str(fileidx)+'_'+context_name+'.npy'
#   out_dir=Path(out_dir)
#   out_dir.mkdir(parents=True, exist_ok=True)
#   #np.save(out_dir / filename, Final_array)#Execution time: 54.30716061592102, 25G
#   filename=str(fileidx)+'_'+'step'+str(step)+'_'+context_name+'.npz'
#   np.savez_compressed(out_dir / filename, Final_array)#Execution time: 579.5185077190399, 7G
#   print(f"Finished np save, Execution time: { time.time() - second_time }") 

if __name__ == "__main__":
  #save validation folders to dict files
  #folders = ["validation_0000","validation_0001","validation_0002","validation_0003","validation_0004","validation_0005","validation_0006","validation_0007"]
  folders = ["validation_0007"]#,"validation_0004"]
  root_path="/DATA5T/Dataset/WaymoDataset/"
  out_dir="/DATA5T/Dataset/Waymodicts/validation"
  data_files = [path for x in folders for path in glob(os.path.join(root_path, x, "*.tfrecord"))]
  print("totoal number of files:", len(data_files))#886
  step=1
  #fileidx=1
  #extract_onesegment_todicts(fileidx, data_files, step, out_dir)
  for fileidx in range(len(data_files)):
      #saveonedictfile(data_files, fileidx, step, out_dir)
      extract_onesegment_todicts(fileidx, data_files, step, out_dir)
  print("finished")
  

    

    # for key, value in framedict.items():
    #   print(key)
# FRONT_BEAM_INCLINATION
# FRONT_LIDAR_EXTRINSIC
# FRONT_RANGE_IMAGE_FIRST_RETURN
# FRONT_RANGE_IMAGE_SECOND_RETURN
# FRONT_CAM_PROJ_FIRST_RETURN
# FRONT_CAM_PROJ_SECOND_RETURN
# REAR_BEAM_INCLINATION
# REAR_LIDAR_EXTRINSIC
# REAR_RANGE_IMAGE_FIRST_RETURN
# REAR_RANGE_IMAGE_SECOND_RETURN
# REAR_CAM_PROJ_FIRST_RETURN
# REAR_CAM_PROJ_SECOND_RETURN
# SIDE_LEFT_BEAM_INCLINATION
# SIDE_LEFT_LIDAR_EXTRINSIC
# SIDE_LEFT_RANGE_IMAGE_FIRST_RETURN
# SIDE_LEFT_RANGE_IMAGE_SECOND_RETURN
# SIDE_LEFT_CAM_PROJ_FIRST_RETURN
# SIDE_LEFT_CAM_PROJ_SECOND_RETURN
# SIDE_RIGHT_BEAM_INCLINATION
# SIDE_RIGHT_LIDAR_EXTRINSIC
# SIDE_RIGHT_RANGE_IMAGE_FIRST_RETURN
# SIDE_RIGHT_RANGE_IMAGE_SECOND_RETURN
# SIDE_RIGHT_CAM_PROJ_FIRST_RETURN
# SIDE_RIGHT_CAM_PROJ_SECOND_RETURN
# TOP_BEAM_INCLINATION
# TOP_LIDAR_EXTRINSIC
# TOP_RANGE_IMAGE_FIRST_RETURN #HxWx6 float32 array with the range image of the first return for this lidar. The six channels are range, intensity, elongation, x, y, and z. The x, y, and z values are in vehicle frame.
# TOP_RANGE_IMAGE_SECOND_RETURN
# TOP_CAM_PROJ_FIRST_RETURN
# TOP_CAM_PROJ_SECOND_RETURN
# FRONT_IMAGE
# FRONT_LEFT_IMAGE
# SIDE_LEFT_IMAGE
# FRONT_RIGHT_IMAGE
# SIDE_RIGHT_IMAGE
# FRONT_INTRINSIC
# FRONT_EXTRINSIC
# FRONT_WIDTH
# FRONT_HEIGHT
# FRONT_LEFT_INTRINSIC
# FRONT_LEFT_EXTRINSIC
# FRONT_LEFT_WIDTH
# FRONT_LEFT_HEIGHT
# FRONT_RIGHT_INTRINSIC
# FRONT_RIGHT_EXTRINSIC
# FRONT_RIGHT_WIDTH
# FRONT_RIGHT_HEIGHT
# SIDE_LEFT_INTRINSIC
# SIDE_LEFT_EXTRINSIC
# SIDE_LEFT_WIDTH
# SIDE_LEFT_HEIGHT
# SIDE_RIGHT_INTRINSIC
# SIDE_RIGHT_EXTRINSIC
# SIDE_RIGHT_WIDTH
# SIDE_RIGHT_HEIGHT
# TOP_RANGE_IMAGE_POSE
# POSE
