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
    smallannotations = []
    valimages = []
    valannotations = []
    smallvalannotations = []
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
        image_dictarraylist=data['images']


        #fetch one image size
        img_width=image_dictarraylist[0]['width']
        img_height=image_dictarraylist[0]['height']
        print(f'image width: {img_width}, height: {img_height}')

        #filter out the annotation
        annotation_dictarray=data['annotations'] #list
        newannotation_dictarray_list=[]
        smallobjectannotation_dictarray_list=[]

        print("annatation_dictarray len",len(annotation_dictarray))
        for annotation_dict in annotation_dictarray:
            globalimgid=annotation_dict['image_id']#global image id
            annotation_dict['iscrowd'] = 0 #add iscrowd
            [x, y, width, height]=annotation_dict['bbox']
            if (width/img_width >0.01 and height/img_height>0.01):
                newannotation_dictarray_list.append(annotation_dict)
            else:
                smallobjectannotation_dictarray_list.append(annotation_dict)

        if fileidx < alltrainfiles_len:
            #annotations.append(annatation_dictarray)
            annotations = np.append(annotations,newannotation_dictarray_list)
            smallannotations = np.append(smallannotations,smallobjectannotation_dictarray_list)
            #images.append(image_dictarray)
            images = np.append(images,image_dictarraylist)
            print("Train Images length:", len(images))
        else:
            valannotations = np.append(valannotations,newannotation_dictarray_list)
            smallvalannotations = np.append(smallvalannotations,smallobjectannotation_dictarray_list)
            valimages = np.append(valimages,image_dictarraylist)
            print("Val Images length:", len(valimages))

    
    print("Annotation type:", type(annotations))
    annotations=annotations.tolist()#Object of type ndarray is not JSON serializable, convert to list first
    smallannotations=smallannotations.tolist()
    print("Annotation type:", type(annotations))
    print("Size of annotation:", len(annotations))
    print("Size of small annotation:", len(smallannotations))
    images=images.tolist()

    valannotations=valannotations.tolist()
    smallvalannotations=smallvalannotations.tolist()
    valimages=valimages.tolist()

    json_out_dir = outputpath #'/DATA5T/Dataset/WaymoCOCO/'#'/data/cmpe249-f20/WaymoCOCOMulti/trainvalall'
    json_out_dir= Path(json_out_dir)
    trainannotationfile = "annotations_train"+str(trainsize)+"filteredbig.json"
    count = 0
    with (json_out_dir / trainannotationfile).open('w') as f:
        for i, anno in enumerate(annotations):
            anno['id'] = i #count #i #set as image frame ID
            count = count +1
        json.dump(dict(images=images, annotations=annotations, categories=categories), f)
        print("Finished json dump train, total count:", count)
    
    trainannotationfile = "annotations_train"+str(trainsize)+"filteredsmall.json"
    count = 0
    with (json_out_dir / trainannotationfile).open('w') as f:
        for i, anno in enumerate(smallannotations):
            anno['id'] = i #count #i #set as image frame ID
            count = count +1
        json.dump(dict(images=images, annotations=smallannotations, categories=categories), f)
        print("Finished json dump train, total small annotation count:", count)

    count=0
    valannotationfile = "annotations_val"+str(valsize)+"filteredbig.json"
    with (json_out_dir / valannotationfile).open('w') as f:
        for i, anno in enumerate(valannotations):
            anno['id'] = i #count #i #set as image frame ID, unique to all other annotations in the dataset
            count = count +1
        json.dump(dict(images=valimages, annotations=valannotations, categories=categories), f)
        print("Finished json dump val, total count:", count)
    
    count=0
    valannotationfile = "annotations_val"+str(valsize)+"filteredsmall.json"
    with (json_out_dir / valannotationfile).open('w') as f:
        for i, anno in enumerate(smallvalannotations):
            anno['id'] = i #count #i #set as image frame ID, unique to all other annotations in the dataset
            count = count +1
        json.dump(dict(images=valimages, annotations=smallvalannotations, categories=categories), f)
        print("Finished json dump val, total count:", count)
    exit()

def convertafilteredimagesubsample(outputpath, allfiles_len, alltrainfiles_len, trainsize, valsize, stepsize):
    images = []
    annotations = []
    smallannotations = []
    valimages = []
    valannotations = []
    smallvalannotations = []
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
        image_dictarraylist=data['images']


        #fetch one image size
        img_width=image_dictarraylist[0]['width']
        img_height=image_dictarraylist[0]['height']
        print(f'image width: {img_width}, height: {img_height}')
        filteredimage_index = {}
        #stepsize =10
        for imagedict in image_dictarraylist:
            globalimage_id=imagedict['id']#000000
            cameraname=imagedict['camera_name']#https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/dataset.proto
            stepcheck=int(globalimage_id) % stepsize
            if cameraname=='1' and stepcheck ==0 :
                print("Image filename: ",imagedict['file_name'])
                print("Image global id:", globalimage_id) #global image id
                if globalimage_id not in filteredimage_index:
                    filteredimage_index[globalimage_id] = []
                    filteredimage_index[globalimage_id].append(imagedict)
        #print(filteredimage_index)

        #filter out the annotation
        annotation_dictarray=data['annotations'] #list
        newannotation_dictarray_list=[]
        smallobjectannotation_dictarray_list=[]

        print("annatation_dictarray len",len(annotation_dictarray))
        for annotation_dict in annotation_dictarray:
            globalimgid=annotation_dict['image_id']#global image id
            if globalimgid in filteredimage_index: #only select images in the list
                annotation_dict['iscrowd'] = 0 #add iscrowd
                [x, y, width, height]=annotation_dict['bbox']
                if (width/img_width >0.01 and height/img_height>0.01):
                    newannotation_dictarray_list.append(annotation_dict)
                else:
                    smallobjectannotation_dictarray_list.append(annotation_dict)

        if fileidx < alltrainfiles_len:
            #annotations.append(annatation_dictarray)
            annotations = np.append(annotations,newannotation_dictarray_list)
            smallannotations = np.append(smallannotations,smallobjectannotation_dictarray_list)
            #images.append(image_dictarray)
            images = np.append(images,image_dictarraylist)
            print("Train Images length:", len(images))
        else:
            valannotations = np.append(valannotations,newannotation_dictarray_list)
            smallvalannotations = np.append(smallvalannotations,smallobjectannotation_dictarray_list)
            valimages = np.append(valimages,image_dictarraylist)
            print("Val Images length:", len(valimages))

    
    print("Annotation type:", type(annotations))
    annotations=annotations.tolist()#Object of type ndarray is not JSON serializable, convert to list first
    smallannotations=smallannotations.tolist()
    print("Annotation type:", type(annotations))
    print("Size of annotation:", len(annotations))
    print("Size of small annotation:", len(smallannotations))
    images=images.tolist()

    valannotations=valannotations.tolist()
    smallvalannotations=smallvalannotations.tolist()
    valimages=valimages.tolist()

    json_out_dir = outputpath #'/DATA5T/Dataset/WaymoCOCO/'#'/data/cmpe249-f20/WaymoCOCOMulti/trainvalall'
    json_out_dir= Path(json_out_dir)
    trainannotationfile = "annotations_train"+str(trainsize)+"filteredbig.json"
    count = 0
    with (json_out_dir / trainannotationfile).open('w') as f:
        for i, anno in enumerate(annotations):
            anno['id'] = i #count #i #set as image frame ID
            count = count +1
        json.dump(dict(images=images, annotations=annotations, categories=categories), f)
        print("Finished json dump train, total count:", count)
    
    trainannotationfile = "annotations_train"+str(trainsize)+"filteredsmall.json"
    count = 0
    with (json_out_dir / trainannotationfile).open('w') as f:
        for i, anno in enumerate(smallannotations):
            anno['id'] = i #count #i #set as image frame ID
            count = count +1
        json.dump(dict(images=images, annotations=smallannotations, categories=categories), f)
        print("Finished json dump train, total small annotation count:", count)

    count=0
    valannotationfile = "annotations_val"+str(valsize)+"filteredbig.json"
    with (json_out_dir / valannotationfile).open('w') as f:
        for i, anno in enumerate(valannotations):
            anno['id'] = i #count #i #set as image frame ID, unique to all other annotations in the dataset
            count = count +1
        json.dump(dict(images=valimages, annotations=valannotations, categories=categories), f)
        print("Finished json dump val, total count:", count)
    
    count=0
    valannotationfile = "annotations_val"+str(valsize)+"filteredsmall.json"
    with (json_out_dir / valannotationfile).open('w') as f:
        for i, anno in enumerate(smallvalannotations):
            anno['id'] = i #count #i #set as image frame ID, unique to all other annotations in the dataset
            count = count +1
        json.dump(dict(images=valimages, annotations=smallvalannotations, categories=categories), f)
        print("Finished json dump val, total count:", count)
    exit()

if __name__ == "__main__":
    allfiles_len=886#len(data_files)
    alltrainfiles_len=684#len(traindata_files)
    print("totoal number of files:", allfiles_len)#886
    print("totoal number of train files:", alltrainfiles_len)#684
    output_dir="/DATA5T/Dataset/WaymoCOCO/"#"/DATA5T/Dataset/WaymoCOCO/pickles"

    create_partial_data = True
    step_size=10
    if create_partial_data:
        #convertasubsample(output_dir, allfiles_len, alltrainfiles_len, 200, 50)
        #convertasubsample(output_dir, allfiles_len, alltrainfiles_len, 20, 5)
        convertafilteredimagesubsample(output_dir, allfiles_len, alltrainfiles_len, alltrainfiles_len, allfiles_len-alltrainfiles_len, step_size)
        #convertasubsample(allfiles_len, alltrainfiles_len, 20, 20)
    