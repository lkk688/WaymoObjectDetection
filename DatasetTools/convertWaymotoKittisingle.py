from os import path as osp
import os
import Waymo2Kitti
from glob import glob

 
root_path="/data/cmpe249-f20/Waymo"
out_dir="/data/cmpe249-f20/WaymoKittiNewSingle"

#folders = ["training_0000","training_0001", "training_0002","training_0003","training_0004","training_0005","training_0006","training_0007","training_0008","training_0009", "training_0010", "training_0015", "training_0016", "training_0017","training_0018", "training_0019", "training_0020", "training_0021","training_0022","training_0023","training_0024","training_0025","training_0026","training_0027","training_0028","training_0029","training_0030","training_0031"]#["training_0001"]# ["training_0000", "training_0001"]
folders = ["training_0000","training_0001", "training_0002","training_0003","training_0004","training_0005","training_0006","training_0007","training_0008","training_0009", "training_0010", "training_0015", "training_0016", "training_0017","training_0018", "training_0019", "training_0020", "training_0021","training_0022","training_0023","training_0024","training_0025","training_0026","training_0027","training_0028","training_0029","training_0030","training_0031","validation_0000","validation_0001","validation_0002","validation_0003","validation_0004","validation_0005","validation_0006","validation_0007"]#["training_0001"]# ["training_0000", "training_0001"]

data_files = [path for x in folders for path in glob(os.path.join(root_path, x, "*.tfrecord"))]
print("totoal number of files:", len(data_files))

workers=50
save_dir = osp.join(out_dir, 'training')#put to the same folder
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
converter = Waymo2Kitti.Waymo2KITTI(
        data_files,
        save_dir,
        workers=workers,
        test_mode=False)
converter.convert_singlethread()
#converter.convert()
   
    

# for i, split in enumerate(folders):
#     #load_dir = osp.join(root_path, 'waymo_format', split)
#     load_dir = osp.join(root_path, split)
#     if split == 'validation':
#         save_dir = osp.join(out_dir, 'validation')
#     else:
#         #save_dir = osp.join(out_dir, 'training', split)
#         save_dir = osp.join(out_dir, 'training')#put to the same folder
#     if not os.path.exists(save_dir):
#         os.makedirs(save_dir)
#     converter = Waymo2Kitti.Waymo2KITTI(
#         load_dir,
#         save_dir,
#         workers=workers,
#         test_mode=(split == 'test'))
#     converter.convert()
