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



def parse_range_image_and_camera_projection(frame):
    """Parse range images and camera projections given a frame.
    Args:
       frame: open dataset frame proto
    Returns:
       range_images: A dict of {laser_name,
         [range_image_first_return, range_image_second_return]}.
       camera_projections: A dict of {laser_name,
         [camera_projection_from_first_return,
          camera_projection_from_second_return]}.
      range_image_top_pose: range image pixel pose for top lidar.
    """
    range_images = {}
    camera_projections = {}
    range_image_top_pose = None
    for laser in frame.lasers:
        if len(laser.ri_return1.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.range_image_compressed, 'ZLIB')
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name] = [ri]

            if laser.name == dataset_pb2.LaserName.TOP:
                range_image_top_pose_str_tensor = tf.io.decode_compressed(
                    laser.ri_return1.range_image_pose_compressed, 'ZLIB')
                range_image_top_pose = dataset_pb2.MatrixFloat()
                range_image_top_pose.ParseFromString(
                    bytearray(range_image_top_pose_str_tensor.numpy()))

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return1.camera_projection_compressed, 'ZLIB')
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name] = [cp]
        if len(laser.ri_return2.range_image_compressed) > 0:  # pylint: disable=g-explicit-length-test
            range_image_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.range_image_compressed, 'ZLIB')
            ri = dataset_pb2.MatrixFloat()
            ri.ParseFromString(bytearray(range_image_str_tensor.numpy()))
            range_images[laser.name].append(ri)

            camera_projection_str_tensor = tf.io.decode_compressed(
                laser.ri_return2.camera_projection_compressed, 'ZLIB')
            cp = dataset_pb2.MatrixInt32()
            cp.ParseFromString(bytearray(camera_projection_str_tensor.numpy()))
            camera_projections[laser.name].append(cp)
    return range_images, camera_projections, range_image_top_pose


def convert_range_image_to_cartesian(frame,
                                     range_images,
                                     range_image_top_pose,
                                     ri_index=0,
                                     keep_polar_features=False):
    """Convert range images from polar coordinates to Cartesian coordinates.
    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
         range_image_second_return]}.
      range_image_top_pose: range image pixel pose for top lidar.
      ri_index: 0 for the first return, 1 for the second return.
      keep_polar_features: If true, keep the features from the polar range image
        (i.e. range, intensity, and elongation) as the first features in the
        output range image.
    Returns:
      dict of {laser_name, (H, W, D)} range images in Cartesian coordinates. D
        will be 3 if keep_polar_features is False (x, y, z) and 6 if
        keep_polar_features is True (range, intensity, elongation, x, y, z).
    """
    cartesian_range_images = {}
    frame_pose = tf.convert_to_tensor(
        value=np.reshape(np.array(frame.pose.transform), [4, 4]))

    # [H, W, 6]
    range_image_top_pose_tensor = tf.reshape(
        tf.convert_to_tensor(value=range_image_top_pose.data),
        range_image_top_pose.shape.dims)
    # [H, W, 3, 3]
    range_image_top_pose_tensor_rotation = transform_utils.get_rotation_matrix(
        range_image_top_pose_tensor[...,
                                    0], range_image_top_pose_tensor[..., 1],
        range_image_top_pose_tensor[..., 2])
    range_image_top_pose_tensor_translation = range_image_top_pose_tensor[..., 3:]
    range_image_top_pose_tensor = transform_utils.get_transform(
        range_image_top_pose_tensor_rotation,
        range_image_top_pose_tensor_translation)

    for c in frame.context.laser_calibrations:
        range_image = range_images[c.name][ri_index]
        if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
            beam_inclinations = range_image_utils.compute_inclination(
                tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
                height=range_image.shape.dims[0])
        else:
            beam_inclinations = tf.constant(c.beam_inclinations)

        beam_inclinations = tf.reverse(beam_inclinations, axis=[-1])
        extrinsic = np.reshape(np.array(c.extrinsic.transform), [4, 4])

        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
        pixel_pose_local = None
        frame_pose_local = None
        if c.name == dataset_pb2.LaserName.TOP:
            pixel_pose_local = range_image_top_pose_tensor
            pixel_pose_local = tf.expand_dims(pixel_pose_local, axis=0)
            frame_pose_local = tf.expand_dims(frame_pose, axis=0)
        range_image_cartesian = range_image_utils.extract_point_cloud_from_range_image(
            tf.expand_dims(range_image_tensor[..., 0], axis=0),
            tf.expand_dims(extrinsic, axis=0),
            tf.expand_dims(tf.convert_to_tensor(
                value=beam_inclinations), axis=0),
            pixel_pose=pixel_pose_local,
            frame_pose=frame_pose_local)

        range_image_cartesian = tf.squeeze(range_image_cartesian, axis=0)

        if keep_polar_features:
            # If we want to keep the polar coordinate features of range, intensity,
            # and elongation, concatenate them to be the initial dimensions of the
            # returned Cartesian range image.
            range_image_cartesian = tf.concat(
                [range_image_tensor[..., 0:3], range_image_cartesian], axis=-1)

        cartesian_range_images[c.name] = range_image_cartesian

    return cartesian_range_images


def convert_range_image_to_point_cloud(frame,
                                       range_images,
                                       camera_projections,
                                       range_image_top_pose,
                                       ri_index=0,
                                       keep_polar_features=False):
    """Convert range images to point cloud.
    Args:
      frame: open dataset frame
      range_images: A dict of {laser_name, [range_image_first_return,
        range_image_second_return]}.
      camera_projections: A dict of {laser_name,
        [camera_projection_from_first_return,
        camera_projection_from_second_return]}.
      range_image_top_pose: range image pixel pose for top lidar.
      ri_index: 0 for the first return, 1 for the second return.
      keep_polar_features: If true, keep the features from the polar range image
        (i.e. range, intensity, and elongation) as the first features in the
        output range image.
    Returns:
      points: {[N, 3]} list of 3d lidar points of length 5 (number of lidars).
        (NOTE: Will be {[N, 6]} if keep_polar_features is true.
      cp_points: {[N, 6]} list of camera projections of length 5
        (number of lidars).
    """
    calibrations = sorted(
        frame.context.laser_calibrations, key=lambda c: c.name)
    points = []
    cp_points = []

    cartesian_range_images = convert_range_image_to_cartesian(
        frame, range_images, range_image_top_pose, ri_index, keep_polar_features)

    for c in calibrations:
        range_image = range_images[c.name][ri_index]
        range_image_tensor = tf.reshape(
            tf.convert_to_tensor(value=range_image.data), range_image.shape.dims)
        range_image_mask = range_image_tensor[..., 0] > 0

        range_image_cartesian = cartesian_range_images[c.name]
        points_tensor = tf.gather_nd(range_image_cartesian,
                                     tf.compat.v1.where(range_image_mask))

        cp = camera_projections[c.name][ri_index]
        cp_tensor = tf.reshape(tf.convert_to_tensor(
            value=cp.data), cp.shape.dims)
        cp_points_tensor = tf.gather_nd(cp_tensor,
                                        tf.compat.v1.where(range_image_mask))
        points.append(points_tensor.numpy())
        cp_points.append(cp_points_tensor.numpy())

    return points, cp_points


def convert_frame_to_dict_lidar(frame):
    """Convert the frame proto into a dict of numpy arrays.
    The keys, shapes, and data types are:
      POSE: 4x4 float32 array
      For each lidar:
        <LIDAR_NAME>_BEAM_INCLINATION: H float32 array
        <LIDAR_NAME>_LIDAR_EXTRINSIC: 4x4 float32 array
        <LIDAR_NAME>_RANGE_IMAGE_FIRST_RETURN: HxWx6 float32 array
        <LIDAR_NAME>_RANGE_IMAGE_SECOND_RETURN: HxWx6 float32 array
        <LIDAR_NAME>_CAM_PROJ_FIRST_RETURN: HxWx6 int64 array
        <LIDAR_NAME>_CAM_PROJ_SECOND_RETURN: HxWx6 float32 array
        (top lidar only) TOP_RANGE_IMAGE_POSE: HxWx6 float32 array

    Args:
      frame: open dataset frame
    Returns:
      Dict from string field name to numpy ndarray.
    """
    # Laser name definition in https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto TOP = 1; FRONT = 2; SIDE_LEFT = 3; SIDE_RIGHT = 4; REAR = 5; The dataset contains data from five lidars - one mid-range lidar (top) and four short-range lidars (front, side left, side right, and rear), ref: https://waymo.com/open/data/perception/ The point cloud of each lidar is encoded as a range image. Two range images are provided for each lidar, one for each of the two strongest returns. It has 4 channels:
    # channel 0: range (see spherical coordinate system definition) channel 1: lidar intensity channel 2: lidar elongation channel 3: is_in_nlz (1 = in, -1 = not in)
    range_images, camera_projection_protos, range_image_top_pose = (
        parse_range_image_and_camera_projection(frame))

    # Convert range images from polar coordinates to Cartesian coordinates
    # dict of {laser_name, (H, W, D)} range images in Cartesian coordinates. D
    # will be 3 if keep_polar_features is False (x, y, z) and 6 if
    # keep_polar_features is True (range, intensity, elongation, x, y, z).
    first_return_cartesian_range_images = convert_range_image_to_cartesian(
        frame, range_images, range_image_top_pose, ri_index=0,
        keep_polar_features=True)

    # second_return_cartesian_range_images = convert_range_image_to_cartesian(
    #     frame, range_images, range_image_top_pose, ri_index=1,
    #     keep_polar_features=True)

    data_dict = {}

    # Save the beam inclinations, extrinsic matrices, first/second return range
    # images, and first/second return camera projections for each lidar.
    for c in frame.context.laser_calibrations:
        laser_name_str = dataset_pb2.LaserName.Name.Name(c.name)

        # beam_inclination_key = f'{laser_name_str}_BEAM_INCLINATION'
        # if len(c.beam_inclinations) == 0:  # pylint: disable=g-explicit-length-test
        #     data_dict[beam_inclination_key] = range_image_utils.compute_inclination(
        #         tf.constant([c.beam_inclination_min, c.beam_inclination_max]),
        #         height=range_images[c.name][0].shape.dims[0]).numpy()
        # else:
        #     data_dict[beam_inclination_key] = np.array(
        #         c.beam_inclinations, np.float32)

        # data_dict[f'{laser_name_str}_LIDAR_EXTRINSIC'] = np.reshape(
        #     np.array(c.extrinsic.transform, np.float32), [4, 4])

        data_dict[f'{laser_name_str}_RANGE_IMAGE_FIRST_RETURN'] = (
            first_return_cartesian_range_images[c.name].numpy())
        # data_dict[f'{laser_name_str}_RANGE_IMAGE_SECOND_RETURN'] = (
        #     second_return_cartesian_range_images[c.name].numpy())

        # first_return_cp = camera_projection_protos[c.name][0]
        # data_dict[f'{laser_name_str}_CAM_PROJ_FIRST_RETURN'] = np.reshape(
        #     np.array(first_return_cp.data), first_return_cp.shape.dims)

        # second_return_cp = camera_projection_protos[c.name][1]
        # data_dict[f'{laser_name_str}_CAM_PROJ_SECOND_RETURN'] = np.reshape(
        #     np.array(second_return_cp.data), second_return_cp.shape.dims)

    # # Save the H x W x 3 RGB image for each camera, extracted from JPEG.
    # for im in frame.images:
    #     cam_name_str = dataset_pb2.CameraName.Name.Name(im.name)
    #     data_dict[f'{cam_name_str}_IMAGE'] = tf.io.decode_jpeg(
    #         im.image).numpy()

    # Save the intrinsics, 4x4 extrinsic matrix, width, and height of each camera.
    # for c in frame.context.camera_calibrations:
    #     cam_name_str = dataset_pb2.CameraName.Name.Name(c.name)
    #     data_dict[f'{cam_name_str}_INTRINSIC'] = np.array(
    #         c.intrinsic, np.float32)
    #     data_dict[f'{cam_name_str}_EXTRINSIC'] = np.reshape(
    #         np.array(c.extrinsic.transform, np.float32), [4, 4])
    #     data_dict[f'{cam_name_str}_WIDTH'] = np.array(c.width)
    #     data_dict[f'{cam_name_str}_HEIGHT'] = np.array(c.height)

    # # Save the range image pixel pose for the top lidar.
    # data_dict['TOP_RANGE_IMAGE_POSE'] = np.reshape(
    #     np.array(range_image_top_pose.data, np.float32),
    #     range_image_top_pose.shape.dims)

    data_dict['POSE'] = np.reshape(
        np.array(frame.pose.transform, np.float32), (4, 4))

    return data_dict


# ref from https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/utils/frame_utils.py
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
    out_dir = Path(save_folder)

    segment_path = tfrecord_pathnames[fileidx]
    c_start = time.time()
    print(
        f'extracting {fileidx}, path: {segment_path}, currenttime: {c_start}')

    dataset = tf.data.TFRecordDataset(str(segment_path), compression_type='')
    # framesdict = {}  # []
    for i, data in enumerate(dataset):
        if i % step != 0:  # Downsample
            continue

        # print('.', end='', flush=True) #progress bar
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        # get one frame
        # A unique name that identifies the frame sequence
        context_name = frame.context.name
        # print('context_name:', context_name)#14824622621331930560_2395_420_2415_420, same to the tfrecord file name
        frame_timestamp_micros = str(frame.timestamp_micros)
        # print(frame_timestamp_micros)
        # context_name as the first level sub folder, frame_timestamp_micros as the second level sub folder
        # one tf record file only has one context_name
        relative_filepath = '/'.join([context_name, frame_timestamp_micros])
        filepath = out_dir / relative_filepath
        filepath.mkdir(parents=True, exist_ok=True)
        # each file os.path.join(input_dir, f'{field}.npy')
        #cameradict = convert_frame_to_dict_cameras(frame)
        lidardict=convert_frame_to_dict_lidar(frame)
        for key, npdata in lidardict.items():#cameradict.items():  # 5 cameras
            filename = key+'.npy'
            completepath = filepath / filename
            np.save(completepath, npdata)
            print(f'Saved {completepath}')
        del frame
        del data
        del lidardict


if __name__ == "__main__":
    # save validation folders to dict files
    folders = ["validation_0000","validation_0001","validation_0002","validation_0003","validation_0004","validation_0005","validation_0006","validation_0007"]
    #folders = ["validation_0005", "validation_0006", "validation_0007"]
    #folders = ["validation_0001","validation_0002","validation_0003","validation_0004","validation_0005","validation_0006","validation_0007"]
    #folders = ["testing_0001","testing_0002","testing_0003","testing_0004","testing_0005","testing_0006","testing_0007"]
    root_path = "/data/cmpe295-liu/Waymo"
    out_dir = "/data/cmpe295-liu/Waymodicts/valdation"
    #out_dir = "/data/cmpe295-liu/Waymodicts/testing"
    data_files = [path for x in folders for path in glob(
        os.path.join(root_path, x, "*.tfrecord"))]
    print("totoal number of files:", len(data_files))  # 886
    step = 1
#   fileidx=1
#   extract_onesegment_todicts(fileidx, data_files, step, out_dir)
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
