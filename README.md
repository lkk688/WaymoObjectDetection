# WaymoObjectDetection

## Waymo Dataset Preparation
* WaymoStartHPC.ipynb is modified based on Waymo official sample and added bounding box visualization of the original Waymo Dataset (TFRecord format)
* create_waymo_train_tfrecord.py and create_waymo_val_tfrecord.py are used to convert the original Waymo Dataset (TFRecord format) to TFRecord files used for Tensorflow object detection
* WaymoNewtoCOCO.ipynb is the code to convert the original Waymo Dataset (TFRecord format) to COCO format.

Use the following code to convert Waymo dataset to Kitti format (it calls converter.concurrenttaskthread()). HPC can open up to 56 (total CPU cores) threads for parallal conversion, 48 threads are tested:
```bash
DatasetTools]$ python Waymo2KittiAsync.py
```
The conversion takes more than 48 hours (the maximum timeout of our HPC GPU node). You can record the finished file index, and continue the conversion from this index via startingindex metric. The converted folder is in 
```bash
(venvpy37cu10) [010796032@g4 DatasetTools]$ ls /data/cmpe249-f20/WaymoKittiAsync/training/
calib  image_0  image_1  image_2  image_3  image_4  label_0  label_1  label_2  label_3  label_4  label_all  pose  velodyne
```

Create train val split file:
```bash
(venvpy37cu10) [010796032@g5 DatasetTools]$ python mycreatewaymoinfo.py --createsplitfile_only
Root path: /data/cmpe249-f20/WaymoKittitMulti/trainall/training
out_dir path: /data/cmpe249-f20/WaymoKittitMulti/trainall/
Total images: 175493
Train size: (140394, 1)
Val size: (35099, 1)
Done in /data/cmpe249-f20/WaymoKittitMulti/trainall/ImageSets/trainval.txt
Done in /data/cmpe249-f20/WaymoKittitMulti/trainall/ImageSets/train.txt
Done in /data/cmpe249-f20/WaymoKittitMulti/trainall/ImageSets/val.txt
```

Use the following code to generate info .pkl files: 
```bash
(venvpy37cu10) [010796032@g5 DatasetTools]$ python mycreatewaymoinfo.py --createinfo_only
Root path: /data/cmpe249-f20/WaymoKittitMulti/trainall/
out_dir path: /data/cmpe249-f20/WaymoKittitMulti/trainall/
Generate info. this may take several minutes.
Generate info. this may take several minutes.
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 140394/140394, 6.9 task/s, elapsed: 20461s, ETA:     0s
Waymo info train file is saved to /data/cmpe249-f20/WaymoKittitMulti/trainall/waymo_infos_train.pkl
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 35099/35099, 8.0 task/s, elapsed: 4376s, ETA:     0s
Waymo info val file is saved to /data/cmpe249-f20/WaymoKittitMulti/trainall/waymo_infos_val.pkl
[>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] 175493/175493, 6.7 task/s, elapsed: 26166s, ETA:     0s
Waymo info trainval file is saved to /data/cmpe249-f20/WaymoKittitMulti/trainall/waymo_infos_trainval.pkl
```

## Object Detection training and evaluation based on Tensorflow2 Object Detection
* Tensorflow2-objectdetection-waymodata.ipynb is the Google Colab sample code to perform object detection and training based on Tensorflow2 object detection (latest version) and utilize the converted Waymo TFRecord file in Google Cloud storage.

## Object Detection training and evaluation based on Pytorch Torchvision (FasterRCNN)
* The sample code to play with Torchvision in Colab: [colab link](https://colab.research.google.com/drive/1DKZUL5ylKjiKtfOCGpirjRA3j8rIOs9M?usp=sharing) (you need to use SJSU google account to view)
* WaymoTrain.ipynb is the sample code to perform object detection training based on Torchvision FasterRCNN based on the original Waymo Dataset (TFRecord format), no dataset conversion is used in this sample code
* WaymoEvaluation.ipynb is the sample code to perform evaluation (both COCO evaluation and image visualization) of the trained object detection model based on Torchvision FasterRCNN
* coco_eval.py, coco_utils.py, engine.py, transforms.py, utils.py are copied from Torchvision directory and used in the WaymoTrain.ipynb and WaymoEvaluation.ipynb

## Object Detection training and evaluation based on Detectron2
* WaymoDetectron2Train.py is the code to run training based on Detectron2. This code used the COCO formated dataset (WaymoDataset converted to COCO via WaymoNewtoCOCO.ipynb)
* WaymoDetectron2Evaluation.ipynb is the jupyter notebook code to run evaluation based on Detectron2

 
