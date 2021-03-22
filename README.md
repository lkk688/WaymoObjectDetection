# WaymoObjectDetection

## Waymo Dataset Preparation
* WaymoStartHPC.ipynb is modified based on Waymo official sample and added bounding box visualization of the original Waymo Dataset (TFRecord format)
* create_waymo_train_tfrecord.py and create_waymo_val_tfrecord.py are used to convert the original Waymo Dataset (TFRecord format) to TFRecord files used for Tensorflow object detection
* WaymoNewtoCOCO.ipynb is the code to convert the original Waymo Dataset (TFRecord format) to COCO format.

Use the following code to convert Waymo dataset to Kitti format:
```bash
DatasetTools]$ python Waymo2KittiAsync.py
```

Use the following code to generate info .pkl files: DatasetTools]$ python mycreatewaymoinfo.py --createinfo_only
![image](https://user-images.githubusercontent.com/6676586/111931827-57952480-8a79-11eb-878c-cab790fca0cd.png)


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

 
