
import sys
from os import path as osp
import numpy as np
print(sys.path)
base_folder='/Developer/3DObject/mmdetection3d/'
# sys.path.insert(0, base_folder)
# print(sys.path)

import mmdet3d
from mmdet3d.apis import init_detector, inference_detector, show_result_meshlab
from mmdet3d.core import Box3DMode

config_file = 'configs/pointpillars/myhv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
checkpoint_file = 'myresults/epoch_120.pth'
point_cloud = '/DATA5T/Dataset/Kitti/testing/velodyne/000000.bin'

config_file = 'configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
checkpoint_file = 'checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth'
point_cloud = base_folder+'demo/kitti_000008.bin'

def boxes3d_to_corners3d(boxes3d, rotate=True):
    """
    :param boxes3d: (N, 7) [x, y, z, h, w, l, ry]
    :param rotate:
    :return: corners3d: (N, 8, 3)
    """
    boxes_num = boxes3d.shape[0]
    h, w, l = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2.], dtype=np.float32).T  # (N, 8)
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T  # (N, 8)

    y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
    y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)

    if rotate:
        ry = boxes3d[:, 6]
        zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
        rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                             [zeros,       ones,       zeros],
                             [np.sin(ry), zeros,  np.cos(ry)]])  # (3, 3, N)
        R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

        temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                       z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
        rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
        x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)

#ref: https://github.com/chris28negu/PointRCNN/blob/master/lib/utils/calibration.py
def corners3d_to_img_boxes(corners3d, P2):
        """
        :param corners3d: (N, 8, 3) corners in rect coordinate
        :return: boxes: (None, 4) [x1, y1, x2, y2] in rgb coordinate
        :return: boxes_corner: (None, 8) [xi, yi] in rgb coordinate
        """
        sample_num = corners3d.shape[0]
        corners3d_hom = np.concatenate((corners3d, np.ones((sample_num, 8, 1))), axis=2)  # (N, 8, 4)

        img_pts = np.matmul(corners3d_hom, P2.T)  # (N, 8, 3), The transposed array.

        x, y = img_pts[:, :, 0] / img_pts[:, :, 2], img_pts[:, :, 1] / img_pts[:, :, 2]
        x1, y1 = np.min(x, axis=1), np.min(y, axis=1)
        x2, y2 = np.max(x, axis=1), np.max(y, axis=1)

        boxes = np.concatenate((x1.reshape(-1, 1), y1.reshape(-1, 1), x2.reshape(-1, 1), y2.reshape(-1, 1)), axis=1)
        boxes_corner = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1)), axis=2)

        return boxes, boxes_corner

# build the model from a config file and a checkpoint file
model = init_detector(base_folder+config_file, base_folder+checkpoint_file, device='cuda:0')

# test a single image and show the results
result, data = inference_detector(model, point_cloud)

points = data['points'][0][0].cpu().numpy()# points number *4
pts_filename = data['img_metas'][0][0]['pts_filename']
file_name = osp.split(pts_filename)[-1].split('.')[0] #006767
print(data['img_metas'])
print(data['img_metas'][0][0]['box_mode_3d']) #Box3DMode.LIDAR

print(len(result))
for res in result[0].keys():
    print(res)
pred_bboxes = result[0]['boxes_3d'].tensor.numpy()
print(pred_bboxes)# [11,7], Each row is (x, y, z, x_size, y_size, z_size, yaw) in Box3DMode.LIDAR
print(type(pred_bboxes))#numpy.ndarray
# boxes_3d
# scores_3d
# labels_3d
corners3d = boxes3d_to_corners3d(pred_bboxes)#11,8,3


if data['img_metas'][0][0]['box_mode_3d'] != Box3DMode.DEPTH:
    points = points[..., [1, 0, 2]]
    points[..., 0] *= -1
    pred_bboxes = Box3DMode.convert(pred_bboxes,
                                    data['img_metas'][0][0]['box_mode_3d'],
                                    Box3DMode.DEPTH)
    print(pred_bboxes)


# visualize the results and save the results in 'results' folder
#model.show_results(data, result, out_dir='results')