from glob import glob
import time
import os
from pathlib import Path
import numpy as np

import visualization_util

import cv2
import datetime
import pickle

category_index = {1: {'id': 1, 'name': 'VEHICLE'},
                  2: {'id': 2, 'name': 'PEDESTRIAN'},
                  3: {'id': 3, 'name': 'SIGN'},
                  4: {'id': 4, 'name': 'CYCLIST'}}

allcameras=["FRONT_IMAGE", "FRONT_LEFT_IMAGE", "FRONT_RIGHT_IMAGE", "SIDE_LEFT_IMAGE", "SIDE_RIGHT_IMAGE"]

if __name__ == "__main__":
    input_data_dir = "/data/cmpe295-liu/Waymodicts/testing/" #"/data/cmpe295-liu/Waymodicts/valdation/"
    context_name = "11096867396355523348_1460_000_1480_000"
    timestamp_micros = "1557240369747551"
    context_dir = os.path.join(input_data_dir, context_name)
    timestamp_dir = os.path.join(context_dir, timestamp_micros)

    nameprefix = "609dtrn2testall"#"610btorchvisiontestall"
    output_dir = "/home/010796032/MyRepo/myoutputs/"+nameprefix+"/"
    dataresult_path = os.path.join(output_dir, context_name, timestamp_micros, 'allcameraresult.npy')
    allcameraresult = np.load(dataresult_path,  allow_pickle=True)
    allcameraresult = allcameraresult.item()

    for imagename in allcameras:#go through all cameras
        # data = {
        #     allcameras[0]: np.load(os.path.join(timestamp_dir, f'{imagename}.npy'))
        # }
        inputimage = np.load(os.path.join(timestamp_dir, f'{imagename}.npy'))

        resultdict=allcameraresult[imagename]#one camera
        #print(f'imagename:{imagename}, resultdict:{resultdict}')
        boxes = resultdict['boxes']
        classes = resultdict['classes']
        scores = resultdict['scores']

        visualization_util.visualize_boxes_and_labels_on_image_array(inputimage, boxes, classes, scores, category_index, use_normalized_coordinates=False,
                                                                     max_boxes_to_draw=200,
                                                                     min_score_thresh=0.1,
                                                                     agnostic_mode=False)
        display_str = f'context_name: {context_name}, timestamp_micros: {timestamp_micros}'
        visualization_util.draw_text_on_image(inputimage, 0, 0, display_str, color='black')

        name = './frame' + str(imagename) + '.jpg'
        print ('Creating\...' + name)
        cv2.imwrite(name, cv2.cvtColor(inputimage, cv2.COLOR_RGB2BGR)) #write to image folder