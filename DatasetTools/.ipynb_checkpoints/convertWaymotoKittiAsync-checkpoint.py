from os import path as osp
import os
import Waymo2KittiAsync
from glob import glob
import asyncio #https://asyncio.readthedocs.io/en/latest/hello_world.html
from waymo_open_dataset import dataset_pb2 as open_dataset
import cv2
import time
import tensorflow as tf
import concurrent.futures

def save_image(image_save_dir,frame, frame_idx):
    """Parse and save the images in png format.

    Args:
        frame (:obj:`Frame`): Open dataset frame proto.
        file_idx (int): Current file index.
        frame_idx (int): Current frame index.
    """
    totalimage_count=0
    for img in frame.images:
        # img_path = f'{self.image_save_dir}{str(img.name - 1)}/' + \
        #     f'{self.prefix}{str(file_idx).zfill(3)}' + \
        #     f'{str(frame_idx).zfill(3)}.png'

        foldername = f'{image_save_dir}{str(img.name - 1)}/'
        img_path = foldername + \
            f'{str(frame_idx).zfill(6)}.png'
        if not os.path.exists(foldername):
            os.makedirs(foldername)
        imag = tf.image.decode_jpeg(img.image).numpy()
        totalimage_count=totalimage_count+1
        cv2.imwrite(str(img_path), cv2.cvtColor(imag, cv2.COLOR_RGB2BGR))
        print("totalimage_count",totalimage_count)
    #await asyncio.sleep(1)

def save_oneimage(img, image_save_dir,frame_idx):
    foldername = f'{image_save_dir}{str(img.name - 1)}/'
    img_path = foldername + \
        f'{str(frame_idx).zfill(6)}.png'
    if not os.path.exists(foldername):
        os.makedirs(foldername)
    imag = tf.image.decode_jpeg(img.image).numpy()
    totalimage_count=totalimage_count+1
    cv2.imwrite(str(img_path), cv2.cvtColor(imag, cv2.COLOR_RGB2BGR))
    print("totalimage_count",totalimage_count)
    
async def imagesave(image_save_dir,frame,frame_idx):
     with concurrent.futures.ProcessPoolExecutor(max_workers=5) as executor:
        futures = {executor.submit(save_oneimage, img, image_save_dir, frame_idx) for img in frame.images }
        for future in concurrent.futures.as_completed(futures):
            print("finished one future")
 
root_path="/data/cmpe249-f20/Waymo"
out_dir="/data/cmpe249-f20/WaymoKittiAsync3"

#folders = ["training_0000","training_0001", "training_0002","training_0003","training_0004","training_0005","training_0006","training_0007","training_0008","training_0009", "training_0010", "training_0015", "training_0016", "training_0017","training_0018", "training_0019", "training_0020", "training_0021","training_0022","training_0023","training_0024","training_0025","training_0026","training_0027","training_0028","training_0029","training_0030","training_0031"]#["training_0001"]# ["training_0000", "training_0001"]
#folders = ["training_0000","training_0001", "training_0002","training_0003","training_0004","training_0005","training_0006","training_0007","training_0008","training_0009", "training_0010", "training_0015", "training_0016", "training_0017","training_0018", "training_0019", "training_0020", "training_0021","training_0022","training_0023","training_0024","training_0025","training_0026","training_0027","training_0028","training_0029","training_0030","training_0031","validation_0000","validation_0001","validation_0002","validation_0003","validation_0004","validation_0005","validation_0006","validation_0007"]#["training_0001"]# ["training_0000", "training_0001"]
folders = ["training_0003"]
data_files = [path for x in folders for path in glob(os.path.join(root_path, x, "*.tfrecord"))]
print("totoal number of files:", len(data_files))

image_save_dir="/data/cmpe249-f20/WaymoKittiAsync3"
start = time.time()
print(start)
file_idx=0
dataset = tf.data.TFRecordDataset(data_files[file_idx], compression_type='')
loop = asyncio.get_event_loop()
for frame_idx, data in enumerate(dataset):
    print(frame_idx)
    c_start = time.time()
    frame = open_dataset.Frame()
    frame.ParseFromString(bytearray(data.numpy()))
    
    future = asyncio.ensure_future(save_image(image_save_dir,frame,frame_idx))
    loop.run_until_complete(future)
    #save_image(image_save_dir, frame, frame_idx)
    print(f"convert one Execution time: { time.time() - c_start }")
    if frame_idx ==10:
        break
print(f"convert all Execution time: { time.time() - start }")
loop.close()
# workers=1
# save_dir = osp.join(out_dir, 'training')#put to the same folder
# if not os.path.exists(save_dir):
#     os.makedirs(save_dir)
# converter = Waymo2KittiAsync.Waymo2KITTIAsync(
#         data_files,
#         save_dir,
#         workers=workers,
#         test_mode=False)
# #converter.convert_singlethread()
# #converter.convert_multithread()
# converter.convertcoroutine()



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
