"""Module to run a simple PointNet on multiple frames of data."""
import numpy as np
#import tensorflow as tf

from . import MyMM3DObjectDetector


# Request the top range image and vehicle pose for each of the 3 most current
# frames.
# DATA_FIELDS = [
#     'TOP_RANGE_IMAGE_FIRST_RETURN',
#     'TOP_RANGE_IMAGE_FIRST_RETURN_1',
#     'TOP_RANGE_IMAGE_FIRST_RETURN_2',
#     'POSE',
#     'POSE_1',
#     'POSE_2',
# ]

# The names of the lidars and input fields that users might want to use for
# detection.
LIDAR_NAMES = ['TOP', 'REAR', 'FRONT', 'SIDE_LEFT', 'SIDE_RIGHT']
LIDAR_FIELD = 'RANGE_IMAGE_FIRST_RETURN'
# The data fields requested from the evaluation script should be specified in
# this field in the module.
DATA_FIELDS = [lidar_name + '_' + LIDAR_FIELD for lidar_name in LIDAR_NAMES]


# Global variables that hold the models and configurations.
model = None

# class kittiargs:
#     modelname = 'kitti'#not used here
#     use_cuda = True
#     basefolder = '/Developer/3DObject/mmdetection3d/'
#     configfile=basefolder+'configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
#     checkpoint = basefolder+ 'myresults/epoch_120.pth'

class kittiargs:
    modelname = 'kitti'#not used here
    use_cuda = True
    basefolder = '/Developer/MyRepo/mymodels/mypointpillar_waymoD5trans_4class/'
    configfile='/home/mymmdetection3d/configs/pointpillars/myhv_pointpillars_secfpn_sbn_2x16_2x_waymo-3d-4class.py'#basefolder+'configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
    checkpoint = basefolder+'epoch_94.pth'

def initialize_model():
  """Method that will be called by the evaluation script to load the model and weights.
  """
  global model
  model = MyMM3DObjectDetector.MyMM3DObjectDetector(kittiargs)


#pc HxWx6 The six channels are range, intensity, elongation, x, y, and z. The x, y, and z values are in vehicle frame.
def transform_point_cloud(pc, old_pose, new_pose):
  """Transform a point cloud from one vehicle frame to another.
  Args:
    pc: N x 6 float32 point cloud, where the final three dimensions are the
        cartesian coordinates of the points in the old vehicle frame.
    old_pose: 4 x 4 float32 vehicle pose at the timestamp of the point cloud
    new_pose: 4 x 4 float32 vehicle pose to transform the point cloud to.
  """
  # Extract the 3x3 rotation matrices and 3x1 translation vectors from the
  # old and new poses.
  # (3, 3)
  old_rot = old_pose[:3, :3]
  # (3, 1)
  old_trans = old_pose[:3, 3:4]
  # (3, 3)
  new_rot = new_pose[:3, :3]
  # (3, 1)
  new_trans = new_pose[:3, 3:4]

  # Extract the local cartesian coordinates from the N x 6 point cloud, adding
  # a new axis at the end to work with np.matmul.
  # (N, 3, 1)
  local_cartesian_coords = pc[..., 3:6][..., np.newaxis]

  # Transform the points from local coordinates to global using the old pose.
  # (N, 3, 1)
  global_coords = old_rot @ local_cartesian_coords + old_trans

  # Transform the points from global coordinates to the new local coordinates
  # using the new pose.
  # (N, 3, 1)
  new_local = np.matrix.transpose(new_rot) @ (global_coords - new_trans)

  # Reassign the dimensions of the range image with the cartesian coordinates
  # in
  pc[..., 3:6] = new_local[..., 0]

def _process_inputs(input_sensor_dict):
  """Converts raw input evaluation data into a dictionary that can be fed into the OpenPCDet model.
  Args:
    input_sensor_dict: dictionary mapping string input data field name to a
    numpy array of the data.
  Returns:
    Dict with pre-processed input that can be passed to the model.
  """
  points = []
  for field in DATA_FIELDS:
    # H x W x 6
    lidar_range_image = input_sensor_dict[field]
    # Flatten and select all points with positive range.
    lidar_range_image = lidar_range_image[lidar_range_image[..., 0] > 0, :]
    # Also transform last dimension from
    #   (range, intensity, elongation, x, y, z) ->
    #   (x, y, z, intensity, elongation)
    points.append(lidar_range_image[:, [3, 4, 5, 1, 2]])

  # Concatenate points from all lidars.
  points_all = np.concatenate(points, axis=0).astype(np.float32)#x*5
  points_all[:, 3] = np.tanh(points_all[:, 3])#??

  return points_all #dataset_processor.prepare_data(data_dict={'points': points_all})


def run_model(**kwargs):
  """Run inference on the pre-loaded OpenPCDet library model.
  Args:
    **kwargs: One keyword argument per input data field from the evaluation
    script.
  Returns:
    Dict from string to numpy ndarray.
  """
  datainput = _process_inputs(kwargs)#x*5 points

  result= model.rundetect(datainput)
  return result

# def run_model2(**kwargs):
#   """Run the model on range images and poses from the most recent 3 frames.
#   Args:
#     **kwargs: Keyword arguments whose names are the entries in DATA_FIELDS.
#   Returns:
#     Dict from string to numpy ndarray.
#   """
#   # (4, 4)
#   newest_pose = kwargs['POSE']
#   # Store the point clouds from each frame.
#   point_clouds = []
#   for suffix in ('', '_1', '_2'):
#     # (H, W, 6)
#     range_image = kwargs['TOP_RANGE_IMAGE_FIRST_RETURN' + suffix]
#     # (N, 6)
#     point_cloud = range_image[range_image[..., 0] > 0]
#     if point_clouds:
#       # (4, 4)
#       src_pose = kwargs['POSE' + suffix]
#       transform_point_cloud(point_cloud, src_pose, newest_pose)
#     point_clouds.append(point_cloud)

#   # Combine the point clouds from all thre frames into a single point cloud.
#   # (N, 6)
#   #combined_pc = tf.concat(point_clouds, axis=0)
#   combined_pc =np.concatenate(point_clouds, axis=0)#numpy alternative

#   # Run the detection model on the combined point cloud.
#   #model = get_model()
#   output_tensors = model(tf.reshape(combined_pc, (1, -1, 6)))

#   # Return the Tensors converted into numpy arrays.
#   return {
#       # Take the first example to go from 1 x B (x 7) to B (x 7).
#       'boxes': output_tensors['boxes'][0, ...].numpy(),
#       'scores': output_tensors['scores'][0, ...].numpy(),
#       # Add a "classes" field that is always CAR.
#       'classes': np.full(output_tensors['boxes'].shape[1], 1, dtype=np.uint8),
#   }