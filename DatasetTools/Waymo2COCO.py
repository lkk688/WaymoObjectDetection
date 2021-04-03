
import os
import cv2
#import tensorflow.compat.v1 as tf
import math
import itertools
import numpy as np
import tensorflow as tf
from glob import glob
from os.path import join
from os import path as osp
from multiprocessing import Pool #https://docs.python.org/3/library/multiprocessing.html
import asyncio #https://asyncio.readthedocs.io/en/latest/hello_world.html
import time
import concurrent.futures


import argparse
from pathlib import Path
import json
from PIL import Image
import sys
import datetime
import pickle

#tf.enable_eager_execution()
try:
    from waymo_open_dataset.utils import range_image_utils
    from waymo_open_dataset.utils import transform_utils
    from waymo_open_dataset.utils import frame_utils
    from waymo_open_dataset import dataset_pb2 as open_dataset
except ImportError:
    raise ImportError(
        'Please run "pip install waymo-open-dataset-tf-2-1-0==1.2.0" '
        'to install the official devkit first.')

# from waymo_open_dataset.utils import range_image_utils, transform_utils
# from waymo_open_dataset.utils.frame_utils import \
#     parse_range_image_and_camera_projection

class Waymo2COCO(object):
    """Waymo to COCO converter.

    This class serves as the converter to change the waymo raw data to KITTI
    format.

    Args:
        alltfrecordfiles: Directory to load waymo raw data.
        save_dir (str): Directory to save data in KITTI format.
        workers (str): Number of workers for the parallel process.
        test_mode (bool): Whether in the test_mode. Default: False.
    """

    def __init__(self,
                 alltfrecordfiles,
                 #load_dir,
                 save_dir,
                 workers=4,
                 startingindex=0,
                 test_mode=False):
        self.filter_empty_3dboxes = True
        self.filter_no_label_zone_points = True

        #self.selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST']
        # self.selected_waymo_classes = ['VEHICLE', 'PEDESTRIAN', 'CYCLIST','SIGN']
        # print("selected_waymo_classes:", self.selected_waymo_classes)

        WAYMO_CLASSES = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']
        self.categories = [{'id': i, 'name': n} for i, n in enumerate(WAYMO_CLASSES)][1:]

        # Only data collected in specific locations will be converted
        # If set None, this filter is disabled
        # Available options: location_sf (main dataset)
        self.selected_waymo_locations = None
        self.save_track_id = False
        self.startingindex = startingindex
        print("startingindex:", startingindex)

        # turn on eager execution for older tensorflow versions
        if int(tf.__version__.split('.')[0]) < 2:
            tf.enable_eager_execution()

        self.lidar_list = [
            '_FRONT', '_FRONT_RIGHT', '_FRONT_LEFT', '_SIDE_RIGHT',
            '_SIDE_LEFT'
        ]
        self.type_list = [
            'UNKNOWN', 'VEHICLE', 'PEDESTRIAN', 'SIGN', 'CYCLIST'
        ]
        self.waymo_to_kitti_class_map = {
            'UNKNOWN': 'DontCare',
            'PEDESTRIAN': 'Pedestrian',
            'VEHICLE': 'Car',
            'CYCLIST': 'Cyclist',
            'SIGN': 'Sign'  # not in kitti
        }

        #self.load_dir = load_dir
        self.save_dir = save_dir
        self.workers = int(workers)
        self.test_mode = test_mode
        self.totalimage_count=0
        self.step=1

        self.tfrecord_pathnames = alltfrecordfiles
        self.totalfilenum =len(self.tfrecord_pathnames)
#         sorted(
#             glob(join(self.load_dir, '*.tfrecord')))

    
    def concurrenttaskthread(self):#https://github.com/jersobh/python-parallelism-examples/blob/master/async_concurrent_futures.py
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.workers) as executor:
            futures = {executor.submit(self.convert_onetfrecord_async, file_idx) for file_idx in range(self.startingindex, self.totalfilenum) }
            for future in concurrent.futures.as_completed(futures):
                #print("finished one future")
                try:
                    print(future.result())
                except:
                    print("Future error.")
        
    async def do_stuff(self,file_idx, frame_idx):
        import random
        await asyncio.sleep(random.uniform(0.1, 0.5)) # NOTE if we hadn't called
            # asyncio.set_event_loop() earlier, we would have to pass an event
            # loop to this function explicitly.
        print("in do stuff: ",file_idx, frame_idx)

    def __len__(self):
        """Length of the filename list."""
        return len(self.tfrecord_pathnames)
    
    def get_camera_labels(self, frame):#Select to use the camera_labels or Lidar 2D labels
        if frame.camera_labels:
            return frame.camera_labels
        return frame.projected_lidar_labels
    
    async def convert_onetfrecord_async(self, file_idx):
        pathname = self.tfrecord_pathnames[file_idx]
        c_start = time.time()
        await self.extract_onesegment_allcamera(file_idx,pathname, self.save_dir, self.step)
        # loop = asyncio.new_event_loop()
        # asyncio.set_event_loop(loop)
        # loop.run_until_complete(self.extract_onesegment_allcamera(file_idx,pathname, self.save_dir, self.step))
        # loop.close()          
        # print(f"convert one async execution time: { time.time() - c_start }")
    
    def convert_multithread(self):
        print('Start multithread converting ...')
        chunksize =1
        process_num=self.workers
        skip_first=False
        tasks = range(len((self.tfrecord_pathnames)))
        task_num = len(tasks)
        pool = Pool(process_num)
        results = []
        gen = pool.imap(self.extract_onesegment_allcamera, tasks, chunksize)
        for result in gen:
            results.append(result)
        pool.close()
        pool.join()

    def extract_onesegment_allcamera(self, fileidx):
        images = []
        annotations = []
        segment_path = self.tfrecord_pathnames[fileidx]
        out_dir=Path(self.save_dir)
        step=self.step
        file_imageglobeid = 0
        imgfoldername='images'
        c_start = time.time()
        #print(c_start)
        global_fileidx= self.startingindex+fileidx
        print(f'extracting {global_fileidx} {fileidx}, path: {segment_path}, currenttime: {c_start}')
        segment_path=Path(segment_path)#convert str to Path object
        segment_name = segment_path.name
        #print(segment_name)
        segment_out_dir = out_dir # / foldername # remove segment_name as one folder, duplicate with image name
        # segment_out_dir = out_dir / segment_name 
        # print(segment_out_dir)#output path + segment_name(with tfrecord)
        segment_out_dir.mkdir(parents=True, exist_ok=True)

        # Creating an empty dictionary for tracking list
        myDict = {}

        dataset = tf.data.TFRecordDataset(str(segment_path), compression_type='')

        for i, data in enumerate(dataset):
            if i % step != 0: #Downsample
                continue

            #print('.', end='', flush=True) #progress bar
            frame = open_dataset.Frame()
            frame.ParseFromString(bytearray(data.numpy()))
            #get one frame

            context_name = frame.context.name #A unique name that identifies the frame sequence
            #print('context_name:', context_name)#14824622621331930560_2395_420_2415_420, same to the tfrecord file name
            frame_timestamp_micros = str(frame.timestamp_micros)

            for index, image in enumerate(frame.images):
                #camera_name = open_dataset.CameraName.Name.Name(image.name)#FRONT, FRONT_LEFT, FRONT_RIGHT, SIDE_LEFT, SIDE_RIGHT 
                camera_name = str(image.name)#

                #image_globeid = image_globeid + 1
                image_globeid = f'{str(global_fileidx).zfill(3)}' + f'{str(file_imageglobeid).zfill(3)}'
                file_imageglobeid = file_imageglobeid + 1
                #print("camera name:", camera_name)

                img = tf.image.decode_jpeg(image.image).numpy()
                image_name='_'.join([frame_timestamp_micros, camera_name])#image name, micros: 1558483937222471
                image_id = '/'.join([context_name, image_name]) #using "/" join, context_name is "14818835630668820137_1780_000_1800_000"
                #New: use sub-folder
                #image_id = '_'.join([context_name, image_name])
                #image_id = '/'.join([context_name, frame_timestamp_micros, camera_name]) #using "/" join
                relative_filepath = '/'.join([imgfoldername, image_id + '.jpg'])
                #print(file_name)
                #filepath = out_dir / file_name
                filepath = segment_out_dir / relative_filepath
                #print('Image output global path:',filepath)
                filepath.parent.mkdir(parents=True, exist_ok=True)

                #images.append(dict(file_name=file_name, id=image_id, height=img.shape[0], width=img.shape[1], camera_name=camera_name))#new add camera_name
                #img_dic=dict(file_name=relative_filepath, id=image_globeid, height=img.shape[0], width=img.shape[1], camera_name=camera_name)
                img_dic=dict(file_name=relative_filepath, id=image_globeid, height=img.shape[0], width=img.shape[1], camera_name=camera_name, context_name=context_name, frame_timestamp_micros=frame_timestamp_micros)
                images.append(img_dic)#new add camera_name
                #print("current image id: ", image_globeid)
                #print("current image dic: ", img_dic)
                #cv2.imwrite(str(filepath), img)
                cv2.imwrite(str(filepath), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

                # for objs in frame.camera_labels: #frame.laser_labels: 
                #     print(objs)
                #     obj=objs[0]
                #     if obj.id in myDict.keys():
                #         imagelist=myDict[obj.id]
                #         imagelist.append(image_globeid)
                #         print(obj)
                #         print('imagelist:',imagelist)
                #     else:
                #         myDict[obj.id]=[image_globeid]

                for camera_labels in self.get_camera_labels(frame):
                    # Ignore camera labels that do not correspond to this camera.
                    if camera_labels.name == image.name:
                        # Iterate over the individual labels.
                        for label in camera_labels.labels:
                            #print("num_lidar_points_in_box: ", label.num_lidar_points_in_box)#always 0
                            #print(label.box)
                            # object bounding box.
                            width = int(label.box.length)
                            height = int(label.box.width)
                            x = int(label.box.center_x - 0.5 * width)
                            y = int(label.box.center_y - 0.5 * height)
                            area = width * height
                            anno_dict = dict(image_id=image_globeid,
                                                    bbox=[x, y, width, height], area=area, category_id=label.type,
                                                    object_id=label.id,
                                                    tracking_difficulty_level=2 if label.tracking_difficulty_level == 2 else 1,
                                                    detection_difficulty_level=2 if label.detection_difficulty_level == 2 else 1)
                            #label.id=4b196757-54a5-4856-b135-3bf5195793ce, a unique id for the object, with globally unique tracking IDs
                            #print("current image annotation dic: ", anno_dict)
                            annotations.append(anno_dict)
                            if label.id in myDict.keys():
                                imagelist=myDict[label.id]
                                imagelist.append(image_globeid)
                            else:
                                myDict[label.id]=[image_globeid]
        
        lasttime=time.time() - c_start
        coco_obj = {'global fileidx':global_fileidx, 'segment_name':segment_name, 'segment_out_dir':segment_out_dir,'num_images':file_imageglobeid, 'lasttime':lasttime, 'images':images, 'annotations':annotations}
        picklefolder= out_dir / 'pickles'
        picklefolder.mkdir(parents=True, exist_ok=True)
        filename='mycocopickle'+str(global_fileidx)+'.pickle'
        picklefilename = picklefolder / filename #'/'.join([picklefolder, 'mycocopickle'+str(global_fileidx)+'.pickle'])
        with open(picklefilename, 'wb') as f:
            pickle.dump(coco_obj, f)
        filename='imagetracklist'+str(global_fileidx)+'.pickle'
        imglistpicklefilename = picklefolder / filename #'/'.join([picklefolder, 'imagetracklist'+str(global_fileidx)+'.pickle'])
        with open(imglistpicklefilename, 'wb') as f:
            pickle.dump(myDict, f)
        #print(f"Finished, Execution time: { time.time() - c_start }")
        print(f'Finished file {global_fileidx}, Execution time: { time.time() - c_start }')
        

    def extract_segment_allcamera(self, tfrecord_files, out_dir, step):

        WAYMO_CLASSES = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']
    
        images = []
        annotations = []
        categories = [{'id': i, 'name': n} for i, n in enumerate(WAYMO_CLASSES)][1:]
        image_globeid=0
        
        for fileidx, segment_path in tfrecord_files:

            print(f'extracting {fileidx}, path: {segment_path}')
            segment_path=Path(segment_path)#convert str to Path object
            segment_name = segment_path.name
            print(segment_name)
            #segment_out_dir = out_dir # remove segment_name as one folder, duplicate with image name
            segment_out_dir = out_dir / segment_name 
            print(segment_out_dir)#output path + segment_name(with tfrecord)
            segment_out_dir.mkdir(parents=True, exist_ok=True)

            dataset = tf.data.TFRecordDataset(str(segment_path), compression_type='')
            
            for i, data in enumerate(dataset):
                if i % step != 0: #Downsample
                    continue

                print('.', end='', flush=True) #progress bar
                frame = open_dataset.Frame()
                frame.ParseFromString(bytearray(data.numpy()))
                #get one frame

                context_name = frame.context.name #A unique name that identifies the frame sequence
                frame_timestamp_micros = str(frame.timestamp_micros)

                for index, image in enumerate(frame.images):
                    camera_name = open_dataset.CameraName.Name.Name(image.name)
                    image_globeid = image_globeid + 1
                    #print("camera name:", camera_name)

                    img = tf.image.decode_jpeg(image.image).numpy()
                    image_name='_'.join([frame_timestamp_micros, camera_name])#image name
                    #image_id = '/'.join([context_name, image_name]) #using "/" join, context_name is the folder
                    #New: use sub-folder
                    image_id = '_'.join([context_name, image_name])
                    #image_id = '/'.join([context_name, frame_timestamp_micros, camera_name]) #using "/" join
                    file_name = image_id + '.jpg'
                    #print(file_name)
                    #filepath = out_dir / file_name
                    filepath = segment_out_dir / file_name
                    print('Image output path:',filepath)
                    filepath.parent.mkdir(parents=True, exist_ok=True)

                    #images.append(dict(file_name=file_name, id=image_id, height=img.shape[0], width=img.shape[1], camera_name=camera_name))#new add camera_name
                    img_dic=dict(file_name=file_name, id=image_globeid, height=img.shape[0], width=img.shape[1], camera_name=camera_name)
                    images.append(img_dic)#new add camera_name
                    print("current image id: ", image_globeid)
                    print("current image dic: ", img_dic)
                    cv2.imwrite(str(filepath), img)

                    for camera_labels in self.get_camera_labels(frame):
                        # Ignore camera labels that do not correspond to this camera.
                        if camera_labels.name == image.name:
                            # Iterate over the individual labels.
                            for label in camera_labels.labels:
                                # object bounding box.
                                width = int(label.box.length)
                                height = int(label.box.width)
                                x = int(label.box.center_x - 0.5 * width)
                                y = int(label.box.center_y - 0.5 * height)
                                area = width * height
                                anno_dict = dict(image_id=image_globeid,
                                                        bbox=[x, y, width, height], area=area, category_id=label.type,
                                                        object_id=label.id,
                                                        tracking_difficulty_level=2 if label.tracking_difficulty_level == 2 else 1,
                                                        detection_difficulty_level=2 if label.detection_difficulty_level == 2 else 1)
                                print("current image annotation dic: ", anno_dict)
                                annotations.append(anno_dict)

        with (segment_out_dir / 'annotations.json').open('w') as f:
            for i, anno in enumerate(annotations):
                anno['id'] = i #set as image frame ID
            json.dump(dict(images=images, annotations=annotations, categories=categories), f)
            

    def save_pose(self, frame, file_idx, frame_idx):
        """Parse and save the pose data.

        Note that SDC's own pose is not included in the regular training
        of KITTI dataset. KITTI raw dataset contains ego motion files
        but are not often used. Pose is important for algorithms that
        take advantage of the temporal information.

        Args:
            frame (:obj:`Frame`): Open dataset frame proto.
            file_idx (int): Current file index.
            frame_idx (int): Current frame index.
        """
        pose = np.array(frame.pose.transform).reshape(4, 4)
        np.savetxt(
            join(f'{self.pose_save_dir}/' +
                 f'{str(file_idx).zfill(3)}{str(frame_idx).zfill(3)}.txt'),
            pose)

    def mkdir_or_exist(self,foldername):
        if not os.path.exists(foldername):
            os.makedirs(foldername)



if __name__ == "__main__":
    folders = ["training_0000","training_0001", "training_0002","training_0003","training_0004","training_0005","training_0006","training_0007","training_0008","training_0009", "training_0010", "training_0015", "training_0016", "training_0017","training_0018", "training_0019", "training_0020", "training_0021","training_0022","training_0023","training_0024","training_0025","training_0026","training_0027","training_0028","training_0029","training_0030","training_0031","validation_0000","validation_0001","validation_0002","validation_0003","validation_0004","validation_0005","validation_0006","validation_0007"]#["training_0001"]# ["training_0000", "training_0001"]
    # folders = ["training_0004"]
    # folder_name="4c_train0004"
    #folders = ["training_0009"]
    folder_name="trainvalall"
    root_path="/data/cmpe249-f20/Waymo"
    out_dir="/data/cmpe249-f20/WaymoCOCOMulti"
    data_files = [path for x in folders for path in glob(os.path.join(root_path, x, "*.tfrecord"))]
    print("totoal number of files:", len(data_files))#886

    workers=32 #16 #48 #16
    startingindex=810 #500 #480 #280
    batch=10
    partialdatafiles = data_files[startingindex:]
    c_start = time.time()
    print(c_start)
    save_dir = osp.join(out_dir, folder_name)
    converter = Waymo2COCO(
            partialdatafiles, #data_files,
            save_dir,
            workers=workers,
            startingindex=startingindex,
            test_mode=False)
    #converter.concurrenttaskthread()#convert_multithread()#convertcoroutine()#concurrenttaskthread()#.convert()
    converter.convert_multithread()
    # for i, split in enumerate(folders):
    #     #load_dir = osp.join(root_path, 'waymo_format', split)
    #     load_dir = osp.join(root_path, split)
    #     if split == 'validation':
    #         save_dir = osp.join(out_dir, 'validation')
    #     else:
    #         save_dir = osp.join(out_dir, 'training', split)
    #     converter = Waymo2KITTIAsync(
    #         data_files,
    #         save_dir,
    #         workers=workers,
    #         test_mode=(split == 'test'))
    #     converter.concurrenttaskthread()#convert_multithread()#convertcoroutine()#concurrenttaskthread()#.convert()
    
    print(f"Finished, Execution time: { time.time() - c_start }")