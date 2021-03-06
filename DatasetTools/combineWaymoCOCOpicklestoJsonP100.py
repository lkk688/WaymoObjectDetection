import itertools
import numpy as np
import pickle
from glob import glob
from os.path import join
from os import path as osp
from pathlib import Path
import json

def convertasubsample(outputpath, allfiles_len, alltrainfiles_len, trainsize, valsize):
    images = []
    annotations = []
    valimages = []
    valannotations = []
    WAYMO_CLASSES =['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist'] #['unknown', 'vehicle', 'pedestrian', 'cyclist']# ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']
    categories = [{'id': i, 'name': n} for i, n in enumerate(WAYMO_CLASSES)][1:]
    pickle_dir = outputpath + "/pickles"

    #allfiles_len=2
    #target_len=alltrainfiles_len #allfiles_len
    for fileidx in range(allfiles_len):
        if fileidx>trainsize and fileidx<alltrainfiles_len:
            continue
        if fileidx>alltrainfiles_len+valsize:
            continue
        print("File idx:", fileidx)
        pickle_filename="mycocopickle"+str(fileidx)+".pickle"
        pickle_filepath=Path(pickle_dir) / pickle_filename
        #print(pickle_filepath)
        # open a file, where you stored the pickled data

        file = open(pickle_filepath, 'rb')
        data = pickle.load(file)
        # close the file
        file.close()
        #images.append(dict(file_name=file_name, id=image_globeid, height=img.shape[0], width=img.shape[1], camera_name=camera_name))
        #img_dic=dict(file_name=file_name, id=image_globeid, height=img.shape[0], width=img.shape[1], camera_name=camera_name)
        print(data['global fileidx'])
        print(data['segment_name'])
        image_dictarray=data['images']
        annatation_dictarray=data['annotations']
        print("annatation_dictarray len",len(annatation_dictarray))
        for annatation_dict in annatation_dictarray:
            annatation_dict['iscrowd'] = 0 #add iscrowd
        if fileidx < alltrainfiles_len:
            #annotations.append(annatation_dictarray)
            annotations = np.append(annotations,annatation_dictarray)
            #images.append(image_dictarray)
            images = np.append(images,image_dictarray)
            print("Train Images length:", len(images))
        else:
            valannotations = np.append(valannotations,annatation_dictarray)
            valimages = np.append(valimages,image_dictarray)
            print("Val Images length:", len(valimages))

    
    print("Annotation type:", type(annotations))
    annotations=annotations.tolist()#Object of type ndarray is not JSON serializable, convert to list first
    print("Annotation type:", type(annotations))
    images=images.tolist()

    valannotations=valannotations.tolist()
    valimages=valimages.tolist()

    json_out_dir = outputpath #'/DATA5T/Dataset/WaymoCOCO/'#'/data/cmpe249-f20/WaymoCOCOMulti/trainvalall'
    json_out_dir= Path(json_out_dir)
    trainannotationfile = "annotations_train"+str(trainsize)+"new.json"
    count = 0
    with (json_out_dir / trainannotationfile).open('w') as f:
        for i, anno in enumerate(annotations):
            anno['id'] = i #count #i #set as image frame ID
            count = count +1
        json.dump(dict(images=images, annotations=annotations, categories=categories), f)
        print("Finished json dump train, total count:", count)
    count=0
    valannotationfile = "annotations_val"+str(valsize)+"new.json"
    with (json_out_dir / valannotationfile).open('w') as f:
        for i, anno in enumerate(valannotations):
            anno['id'] = i #count #i #set as image frame ID, unique to all other annotations in the dataset
            count = count +1
        json.dump(dict(images=valimages, annotations=valannotations, categories=categories), f)
        print("Finished json dump val, total count:", count)
    exit()


if __name__ == "__main__":
    allfiles_len=886#len(data_files)
    alltrainfiles_len=684#len(traindata_files)
    print("totoal number of files:", allfiles_len)#886
    print("totoal number of train files:", alltrainfiles_len)#684
    output_dir="/DATA5T/Dataset/WaymoCOCO/"#"/DATA5T/Dataset/WaymoCOCO/pickles"

    create_partial_data = True
    if create_partial_data:
        convertasubsample(output_dir, allfiles_len, alltrainfiles_len, 200, 50)
        #convertasubsample(allfiles_len, alltrainfiles_len, 20, 20)
    