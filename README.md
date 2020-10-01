# WaymoObjectDetection

## Waymo Dataset Preparation
* WaymoStartHPC.ipynb is modified based on Waymo official sample and added bounding box visualization of the original Waymo Dataset (TFRecord format)
* create_waymo_train_tfrecord.py and create_waymo_val_tfrecord.py are used to convert the original Waymo Dataset (TFRecord format) to TFRecord files used for Tensorflow object detection
* WaymoNewtoCOCO.ipynb is the code to convert the original Waymo Dataset (TFRecord format) to COCO format.

## Object Detection training and evaluation based on Tensorflow2 Object Detection
* Tensorflow2-objectdetection-waymodata.ipynb is the Google Colab sample code to perform object detection and training based on Tensorflow2 object detection (latest version) and utilize the converted Waymo TFRecord file in Google Cloud storage.

## Object Detection training and evaluation based on Pytorch Torchvision (FasterRCNN)
* WaymoTrain.ipynb is the sample code to perform object detection training based on Torchvision FasterRCNN based on the original Waymo Dataset (TFRecord format), no dataset conversion is used in this sample code
* WaymoEvaluation.ipynb is the sample code to perform evaluation (both COCO evaluation and image visualization) of the trained object detection model based on Torchvision FasterRCNN

 
