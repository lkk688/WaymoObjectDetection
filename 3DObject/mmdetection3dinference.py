
from My3DObjectDetector import MyMM3DObjectDetector
import glob
import time
import pickle

config_file = 'configs/pointpillars/myhv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
checkpoint_file = 'myresults/epoch_120.pth'
point_cloud = '/DATA5T/Dataset/Kitti/testing/velodyne/000000.bin'

config_file = 'configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
checkpoint_file = 'checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth'
point_cloud = 'demo/kitti_000008.bin'

class mm3ddetectorargs:
    modelname = 'second'#not used here
    use_cuda = True
    basefolder = '/Developer/3DObject/mmdetection3d/'
    configfile=basefolder+'configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
    checkpoint = basefolder+ 'checkpoints/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class_20200620_230421-aa0f3adb.pth'

class kittiargs:
    modelname = 'kitti'#not used here
    use_cuda = True
    basefolder = '/Developer/3DObject/mmdetection3d/'
    configfile=basefolder+'configs/pointpillars/hv_pointpillars_secfpn_6x8_160e_kitti-3d-3class.py'
    checkpoint = basefolder+ 'myresults/epoch_120.pth'

mydetector = MyMM3DObjectDetector(kittiargs)#mm3ddetectorargs)

#point_cloud = '/Developer/3DObject/mmdetection3d/demo/kitti_000008.bin'
point_cloud = '/DATA5T/Dataset/Kitti/testing/velodyne/000001.bin'

point_cloud_path = '/DATA5T/Dataset/Kitti/testing/velodyne'
lidarpath=sorted(glob.glob(point_cloud_path+'/*.bin'))
lidarlen=len(lidarpath)
print("Total lidar:", lidarlen)

results=[]
t_start = time.time()
for lidaridx in range(0,200):
    c_start = time.time()
    point_cloud = lidarpath[lidaridx]
    result= mydetector.detect(point_cloud)
    boxes_3d = result[0]['boxes_3d'].tensor.numpy()
    scores_3d = result[0]['scores_3d'].numpy()
    labels_3d = result[0]['labels_3d'].numpy()
    print(labels_3d)
    lasttime=time.time() - c_start
    result_obj = {'lidaridx':lidaridx, 'point_cloud_path':point_cloud, 'boxes_3d':boxes_3d,'scores_3d':scores_3d, 'labels_3d':labels_3d, 'lasttime':lasttime}
    results.append(result_obj)
    print(f'Finished file {lidaridx}, Execution time: { lasttime }')
        
picklefilename = 'testkitti.pickle'
with open(picklefilename, 'wb') as f:
    pickle.dump(results, f)
print(f"Finished save to {picklefilename}, Execution time: { time.time() - t_start }")
        

# print(len(result))
# for res in result[0].keys():
#     print(res)
# # boxes_3d
# # scores_3d
# # labels_3d

# print(pred_bboxes)# [11,7], Each row is (x, y, z, x_size, y_size, z_size, yaw) in Box3DMode.LIDAR
# print(type(pred_bboxes))#numpy.ndarray




