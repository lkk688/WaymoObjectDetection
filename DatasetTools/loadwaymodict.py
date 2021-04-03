from glob import glob
import time
import os
from pathlib import Path
import numpy as np

if __name__ == "__main__":
    #test the above functions: convert a Frame proto into a dictionary
    #convert_frame_to_dict
    base_dir="/data/cmpe249-f20/WaymoKittitMulti/dict_train0"
    base_dir = Path(base_dir)
    filename="1_step10_10017090168044687777_6380_000_6400_000.npz"

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
    print(type(framedict['TOP_RANGE_IMAGE_FIRST_RETURN']))
    print(framedict['TOP_RANGE_IMAGE_FIRST_RETURN'].shape)
