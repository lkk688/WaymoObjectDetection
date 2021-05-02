import itertools
import numpy as np
import pickle
from glob import glob
from os.path import join
from os import path as osp
from pathlib import Path
import json

def convertasubsample(allfiles_len, alltrainfiles_len, trainsize, valsize):
    images = []
    annotations = []
    valimages = []
    valannotations = []
    WAYMO_CLASSES =['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist'] #['unknown', 'vehicle', 'pedestrian', 'cyclist']# ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']
    categories = [{'id': i, 'name': n} for i, n in enumerate(WAYMO_CLASSES)][1:]

    #allfiles_len=2
    target_len=alltrainfiles_len #allfiles_len
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

    json_out_dir = '/data/cmpe249-f20/WaymoCOCOMulti/trainvalall'
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
    folders = ["training_0000","training_0001", "training_0002","training_0003","training_0004","training_0005","training_0006","training_0007","training_0008","training_0009", "training_0010", "training_0015", "training_0016", "training_0017","training_0018", "training_0019", "training_0020", "training_0021","training_0022","training_0023","training_0024","training_0025","training_0026","training_0027","training_0028","training_0029","training_0030","training_0031","validation_0000","validation_0001","validation_0002","validation_0003","validation_0004","validation_0005","validation_0006","validation_0007"]#["training_0001"]# ["training_0000", "training_0001"]
    
    trainfolders = ["training_0000","training_0001", "training_0002","training_0003","training_0004","training_0005","training_0006","training_0007","training_0008","training_0009", "training_0010", "training_0015", "training_0016", "training_0017","training_0018", "training_0019", "training_0020", "training_0021","training_0022","training_0023","training_0024","training_0025","training_0026","training_0027","training_0028","training_0029","training_0030","training_0031"]#["training_0001"]# ["training_0000", "training_0001"]
    valfolders = ["validation_0000","validation_0001","validation_0002","validation_0003","validation_0004","validation_0005","validation_0006","validation_0007"]
    # folders = ["training_0004"]
    # folder_name="4c_train0004"
    #folders = ["training_0009"]
    folder_name="trainvalall"
    root_path="/data/cmpe249-f20/Waymo"
    pickle_dir="/data/cmpe249-f20/WaymoCOCOMulti/trainvalall/pickles"
    data_files = [path for x in folders for path in glob(join(root_path, x, "*.tfrecord"))]
    traindata_files = [path for x in trainfolders for path in glob(join(root_path, x, "*.tfrecord"))]
    allfiles_len=len(data_files)
    alltrainfiles_len=len(traindata_files)
    print("totoal number of files:", allfiles_len)#886
    print("totoal number of train files:", alltrainfiles_len)#886

    create_partial_data = True
    if create_partial_data:
        convertasubsample(allfiles_len, alltrainfiles_len, 20, 20)

    images = []
    annotations = []
    valimages = []
    valannotations = []
    WAYMO_CLASSES = ['unknown', 'vehicle', 'pedestrian', 'sign', 'cyclist']
    categories = [{'id': i, 'name': n} for i, n in enumerate(WAYMO_CLASSES)][1:]

    #allfiles_len=2
    target_len=alltrainfiles_len #allfiles_len
    for fileidx in range(allfiles_len):
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

    json_out_dir = '/data/cmpe249-f20/WaymoCOCOMulti/trainvalall'
    json_out_dir= Path(json_out_dir)
    count = 0
    with (json_out_dir / 'annotations_trainallnew.json').open('w') as f:
        for i, anno in enumerate(annotations):
            anno['id'] = i #count #i #set as image frame ID
            count = count +1
        json.dump(dict(images=images, annotations=annotations, categories=categories), f)
        print("Finished json dump train, total count:", count)
    count=0
    with (json_out_dir / 'annotations_valallnew.json').open('w') as f:
        for i, anno in enumerate(valannotations):
            anno['id'] = i #count #i #set as image frame ID
            count = count +1
        json.dump(dict(images=valimages, annotations=valannotations, categories=categories), f)
        print("Finished json dump val, total count:", count)


        # cnt = 0
        # #coco_obj = {'global fileidx':global_fileidx, 'segment_name':segment_name, 'segment_out_dir':segment_out_dir,'num_images':file_imageglobeid, 'lasttime':lasttime, 'images':images, 'annotations':annotations}
        # for item in data:
        #     print('The data ', cnt, ' is : ', item)
        #     cnt += 1


