#generated dict files in /home/010796032/MyRepo/WaymoObjectDetection/DatasetTools/waymoframe2dict.py
#dict file path: /data/cmpe249-f20/WaymoKittitMulti/validationalldicts
#test loading one dict file: /home/010796032/MyRepo/WaymoObjectDetection/DatasetTools/loadwaymodict.py

from glob import glob
import time
import os
from pathlib import Path
import numpy as np
import cv2
import datetime
import pickle

def loadonedictfile(filename):
    Final_array=np.load(base_dir / filename, allow_pickle=True, mmap_mode='r')
    data_array=Final_array['arr_0']
    array_len=len(data_array)
    print("Final_array lenth:", array_len)
    print("Final_array type:", type(data_array))

    #for frameid in range(array_len):
    frameid=0
    print("frameid:", frameid)
    convertedframesdict = data_array[frameid] #{'key':key, 'context_name':context_name, 'framedict':framedict}
    frame_timestamp_micros=convertedframesdict['key']
    context_name=convertedframesdict['context_name']
    framedict=convertedframesdict['framedict']
    print('context_name:', context_name)
    print('frame_timestamp_micros:', frame_timestamp_micros)
    for key, value in framedict.items():
        print(key)
    print(type(framedict['FRONT_IMAGE']))
    print(framedict['FRONT_IMAGE'].shape)

if __name__ == "__main__":
    #test the above functions: convert a Frame proto into a dictionary
    #convert_frame_to_dict
    base_dir="/data/cmpe249-f20/WaymoKittitMulti/validationalldicts"
    base_dir = Path(base_dir)
    filename="0_step1_10203656353524179475_7625_000_7645_000.npz"

    loadonedictfile(filename)
