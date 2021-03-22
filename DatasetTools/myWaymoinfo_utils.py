import numpy as np
from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path
from skimage import io
import mmcv
#from mmdet3d.core.bbox import box_np_ops
import box_np_ops

def get_image_index_str(img_idx, use_prefix_id=False):
    if use_prefix_id: #Our converted data does not have prefix
        return '{:07d}'.format(img_idx)
    else:
        return '{:06d}'.format(img_idx)

def get_waymo_info_path(idx,
                        prefixpath,
                        info_type='image_2',
                        file_tail='.png',
                        exist_check=True,
                        use_prefix_id=False):#Our converted data does not have prefix
    img_idx_str = get_image_index_str(idx, use_prefix_id)
    img_idx_str += file_tail
    prefixpath = Path(prefixpath)
    file_path = prefixpath / info_type / img_idx_str
    if exist_check and not (file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    return str(file_path)

def get_waymo_info_relativepath(idx,
                        training,
                        prefixpath, #only used to check exist or not
                        info_type='image_2',
                        file_tail='.png',
                        exist_check=True,
                        use_prefix_id=False):#Our converted data does not have prefix
    img_idx_str = get_image_index_str(idx, use_prefix_id)
    img_idx_str += file_tail
    if training:
        relative_file_path = Path('training') / info_type / img_idx_str
    else:
        #file_path = Path('testing') / info_type / img_idx_str
        relative_file_path = Path('test') / info_type / img_idx_str
    #relative_file_path =  Path(info_type) / img_idx_str
    if exist_check and not (prefixpath / relative_file_path).exists():
        raise ValueError('file not exist: {}'.format(relative_file_path))
    return str(relative_file_path)

def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'alpha': [],
        'bbox': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])
    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    annotations['alpha'] = np.array([float(x[3]) for x in content])
    annotations['bbox'] = np.array([[float(info) for info in x[4:8]]
                                    for x in content]).reshape(-1, 4)
    # dimensions will convert hwl format to standard lhw(camera) format.
    annotations['dimensions'] = np.array([[float(info) for info in x[8:11]]
                                          for x in content
                                          ]).reshape(-1, 3)[:, [2, 0, 1]]
    annotations['location'] = np.array([[float(info) for info in x[11:14]]
                                        for x in content]).reshape(-1, 3)
    annotations['rotation_y'] = np.array([float(x[14])
                                          for x in content]).reshape(-1)
    if len(content) != 0 and len(content[0]) == 16:  # have score
        annotations['score'] = np.array([float(x[15]) for x in content])
    else:
        annotations['score'] = np.zeros((annotations['bbox'].shape[0], ))
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    annotations['group_ids'] = np.arange(num_gt, dtype=np.int32)
    return annotations  

def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

def add_difficulty_to_annos(info):
    min_height = [40, 25,
                  25]  # minimum height for evaluated groundtruth/detections
    max_occlusion = [
        0, 1, 2
    ]  # maximum occlusion level of the groundtruth used for evaluation
    max_trunc = [
        0.15, 0.3, 0.5
    ]  # maximum truncation level of the groundtruth used for evaluation
    annos = info['annos']
    dims = annos['dimensions']  # lhw format
    bbox = annos['bbox']
    height = bbox[:, 3] - bbox[:, 1]
    occlusion = annos['occluded']
    truncation = annos['truncated']
    diff = []
    easy_mask = np.ones((len(dims), ), dtype=np.bool)
    moderate_mask = np.ones((len(dims), ), dtype=np.bool)
    hard_mask = np.ones((len(dims), ), dtype=np.bool)
    i = 0
    for h, o, t in zip(height, occlusion, truncation):
        if o > max_occlusion[0] or h <= min_height[0] or t > max_trunc[0]:
            easy_mask[i] = False
        if o > max_occlusion[1] or h <= min_height[1] or t > max_trunc[1]:
            moderate_mask[i] = False
        if o > max_occlusion[2] or h <= min_height[2] or t > max_trunc[2]:
            hard_mask[i] = False
        i += 1
    is_easy = easy_mask
    is_moderate = np.logical_xor(easy_mask, moderate_mask)
    is_hard = np.logical_xor(hard_mask, moderate_mask)

    for i in range(len(dims)):
        if is_easy[i]:
            diff.append(0)
        elif is_moderate[i]:
            diff.append(1)
        elif is_hard[i]:
            diff.append(2)
        else:
            diff.append(-1)
    annos['difficulty'] = np.array(diff, np.int32)
    return diff

def get_waymo_image_info(path,
                         training=True,
                         label_info=True,
                         velodyne=False,
                         calib=False,
                         pose=False,
                         image_ids=7481,# image id list
                         extend_matrix=True,
                         num_worker=8,
                         relative_path=True,
                         with_imageshape=True,
                         max_sweeps=5):
    """
    Waymo annotation format version like KITTI:
    {
        [optional]points: [N, 3+] point cloud
        [optional, for kitti]image: {
            image_idx: ...
            image_path: ...
            image_shape: ...
        }
        point_cloud: {
            num_features: 4 #6
            velodyne_path: ...
        }
        [optional, for kitti]calib: {
            R0_rect: ...
            Tr_velo_to_cam0: ...
            P0: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
            [optional]difficulty: kitti difficulty
            [optional]group_ids: used for multi-part object
        }
    }
    """
    root_path = Path(path)
    if not isinstance(image_ids, list): # image id list in train.txt
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}
        pc_info = {'num_features': 4} #6
        calib_info = {}

        image_info = {'image_idx': idx}
        annotations = None
        if velodyne:
            pc_info['velodyne_path'] = get_waymo_info_relativepath(idx,training,path,'velodyne','.bin')
            #info['timestamp'] = np.int64(points[0, -1]) # our conversion does not have the timestamp
            # values of the last dim are all the timestamp
        image_info['image_path'] = get_waymo_info_relativepath(idx,training,path,'image_0','.png')
        
        if with_imageshape:
            img_path = image_info['image_path']
            if relative_path:
                img_path = str(root_path / img_path) #convert to absolute path
            image_info['image_shape'] = np.array(
                io.imread(img_path).shape[:2], dtype=np.int32)# get image shape
        if label_info:
            label_path = get_waymo_info_relativepath(idx,training,path,'label_all','.txt')
            if relative_path:#convert to absolute path
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)
        info['image'] = image_info
        info['point_cloud'] = pc_info
        if calib:
            calib_path = get_waymo_info_relativepath(idx,training,path,'calib','.txt')
            if relative_path:#convert to absolute path
                calib_path = str(root_path / calib_path)
            with open(calib_path, 'r') as f:
                lines = f.readlines()
            P0 = np.array([float(info) for info in lines[0].split(' ')[1:13]
                           ]).reshape([3, 4])
            P1 = np.array([float(info) for info in lines[1].split(' ')[1:13]
                           ]).reshape([3, 4])
            P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]
                           ]).reshape([3, 4])
            P3 = np.array([float(info) for info in lines[3].split(' ')[1:13]
                           ]).reshape([3, 4])
            P4 = np.array([float(info) for info in lines[4].split(' ')[1:13]
                           ]).reshape([3, 4])
            if extend_matrix:
                P0 = _extend_matrix(P0)
                P1 = _extend_matrix(P1)
                P2 = _extend_matrix(P2)
                P3 = _extend_matrix(P3)
                P4 = _extend_matrix(P4)
            R0_rect = np.array([
                float(info) for info in lines[5].split(' ')[1:10]
            ]).reshape([3, 3])
            if extend_matrix:
                rect_4x4 = np.zeros([4, 4], dtype=R0_rect.dtype)
                rect_4x4[3, 3] = 1.
                rect_4x4[:3, :3] = R0_rect
            else:
                rect_4x4 = R0_rect

            Tr_velo_to_cam = np.array([
                float(info) for info in lines[6].split(' ')[1:13]
            ]).reshape([3, 4])
            if extend_matrix:
                Tr_velo_to_cam = _extend_matrix(Tr_velo_to_cam)
            calib_info['P0'] = P0
            calib_info['P1'] = P1
            calib_info['P2'] = P2
            calib_info['P3'] = P3
            calib_info['P4'] = P4
            calib_info['R0_rect'] = rect_4x4
            calib_info['Tr_velo_to_cam'] = Tr_velo_to_cam #only for camera 0
            info['calib'] = calib_info
        if pose:
            pose_path = get_waymo_info_relativepath(idx,training,path,'pose','.txt')
            if relative_path:#convert to absolute path
                pose_path = str(root_path / pose_path)
            info['pose'] = np.loadtxt(pose_path)

        if annotations is not None:
            info['annos'] = annotations
            info['annos']['camera_id'] = info['annos'].pop('score')
            add_difficulty_to_annos(info)

        sweeps = []
        prev_idx = idx
        while len(sweeps) < max_sweeps:
            prev_info = {}
            prev_idx -= 1
            prev_info['velodyne_path'] = get_waymo_info_relativepath(prev_idx,training,path,'velodyne','.bin', exist_check=False)
            lidar_path = prev_info['velodyne_path']
            if relative_path:#convert to absolute path
                lidar_path = str(root_path / lidar_path)
            if_prev_exists = osp.exists(lidar_path)
            if if_prev_exists:
                prev_points = np.fromfile(
                    Path(lidar_path), dtype=np.float32)
                prev_points = np.copy(prev_points).reshape(
                    -1, pc_info['num_features'])
                #prev_info['timestamp'] = np.int64(prev_points[0, -1])
                prev_pose_path = get_waymo_info_relativepath(prev_idx,training,path,'pose','.txt')
                if relative_path:#convert to absolute path
                    prev_pose_path = str(root_path / prev_pose_path)
                prev_info['pose'] = np.loadtxt(prev_pose_path)
                sweeps.append(prev_info)
            else:
                break
        info['sweeps'] = sweeps

        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)

    return list(image_infos)


def _calculate_num_points_in_gt(data_path,
                                infos,
                                relative_path,
                                remove_outside=True,
                                num_features=4):
    for info in mmcv.track_iter_progress(infos):
        pc_info = info['point_cloud']
        image_info = info['image']
        calib = info['calib']
        if relative_path:
            v_path = str(Path(data_path) / pc_info['velodyne_path'])
        else:
            v_path = pc_info['velodyne_path']
        points_v = np.fromfile(
            v_path, dtype=np.float32, count=-1).reshape([-1, num_features])
        rect = calib['R0_rect']
        Trv2c = calib['Tr_velo_to_cam']
        P2 = calib['P2']
        if remove_outside:
            points_v = box_np_ops.remove_outside_points(
                points_v, rect, Trv2c, P2, image_info['image_shape'])

        # points_v = points_v[points_v[:, 0] > 0]
        annos = info['annos']
        num_obj = len([n for n in annos['name'] if n != 'DontCare'])
        # annos = kitti.filter_kitti_anno(annos, ['DontCare'])
        dims = annos['dimensions'][:num_obj]
        loc = annos['location'][:num_obj]
        rots = annos['rotation_y'][:num_obj]
        gt_boxes_camera = np.concatenate([loc, dims, rots[..., np.newaxis]],
                                         axis=1)
        gt_boxes_lidar = box_np_ops.box_camera_to_lidar(
            gt_boxes_camera, rect, Trv2c)
        indices = box_np_ops.points_in_rbbox(points_v[:, :3], gt_boxes_lidar)
        num_points_in_gt = indices.sum(0)
        num_ignored = len(annos['dimensions']) - num_obj
        num_points_in_gt = np.concatenate(
            [num_points_in_gt, -np.ones([num_ignored])])
        annos['num_points_in_gt'] = num_points_in_gt.astype(np.int32)