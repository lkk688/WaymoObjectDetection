
from My3DObjectDetector import MyMM3DObjectDetector

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

mydetector = MyMM3DObjectDetector(mm3ddetectorargs)



