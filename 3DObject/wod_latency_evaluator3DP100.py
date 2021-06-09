# from https://github.com/waymo-research/waymo-open-dataset/blob/master/waymo_open_dataset/latency/wod_latency_evaluator.py
"""The WOD (Waymo Open Dataset) latency evaluation.
This script runs a user-submitted detection model on WOD data (extracted into
a set of numpy arrays), saves the outputs of those models as numpy arrays, and
records the latency of the detection model's inference. Both the input data and
the output detection results are stored in the following directory structure:
`<root_dir>/<context.name>/<timestamp_micros>/` with one such subdirectory for
every frame. The input data is stored in a series of numpy files where the
basename is the name of the field; see convert_frame_to_dict in
utils/frame_utils.py for a list of the field names. The detection results are
saved in three numpy files: boxes, scores, and classes. Finally, the latency
results are saved in a text file containing the latency values for each frame.
NOTE: This is a standalone script that does not depend on any TensorFlow,
PyTorch, or WOD code, or on bazel. It is intended to run in a docker container
submitted by the user, with the only requirements being a numpy installation and
a user-generated wod_latency_submission module.
"""
import argparse
import os
import time

import numpy as np

import wod_latency_submission


def process_example(input_dir, output_dir):
    """Process a single example, save its outputs, and return the latency.
    In particular, this function requires that the submitted model (the run_model
    function in the wod_latency_submission module) takes in a bunch of numpy
    arrays as keyword arguments, whose names are defined in
    wod_latency_submission.DATA_FIELDS. It also assumes that input_directory
    contains a bunch of .npy data files with basenames corresponding to the valid
    values of DATA_FIELDS. Thus, this function loads the required fields from the
    input_directory, runs and times the run_model function, saves the model's
    outputs (a 'boxes' N x 7 ndarray, a 'scores' N ndarray, and a 'classes' N
    ndarray) to the output directory, and returns the model's runtime in seconds.
    Args:
      input_dir: string directory name to find the input .npy data files
      output_dir: string directory name to save the model results to.
    Returns:
      float latency value of the run_model call, in seconds.
    """
    # Load all the data fields that the user requested.
    data = {
        field: np.load(os.path.join(input_dir, f'{field}.npy'))
        for field in wod_latency_submission.DATA_FIELDS
    }

    # Time the run_model function of the user's submitted module, with the data
    # fields passed in as keyword arguments.
    tic = time.perf_counter()
    output = wod_latency_submission.run_model(**data)
    toc = time.perf_counter()

    # Sanity check the output before saving.
    assert len(output) == 3
    assert set(output.keys()) == set(('boxes', 'scores', 'classes'))
    num_objs = output['boxes'].shape[0]
    print(f'num_objs:{num_objs}')
    assert output['scores'].shape[0] == num_objs
    assert output['classes'].shape[0] == num_objs

    # Save the outputs as numpy files.
    for k, v in output.items():
        np.save(os.path.join(output_dir, k), v)

    # Save the list of input fields in a text file.
    with open(os.path.join(output_dir, 'input_fields.txt'), 'w') as f:
        f.write('\n'.join(wod_latency_submission.DATA_FIELDS))

    # Return the elapsed time of the run_model call.
    return toc - tic

allcameras=["FRONT_IMAGE", "FRONT_LEFT_IMAGE", "FRONT_RIGHT_IMAGE", "SIDE_LEFT_IMAGE", "SIDE_RIGHT_IMAGE"]
def process_allimages_example(input_dir, output_dir):
    # Load all the data fields that the user requested.
    latency=[]
    result_dict={}
    for imagename in allcameras:#go through all cameras
        data = {
            allcameras[0]: np.load(os.path.join(input_dir, f'{imagename}.npy'))
        }

        # Time the run_model function of the user's submitted module, with the data
        # fields passed in as keyword arguments.
        tic = time.perf_counter()
        output = wod_latency_submission.run_model(**data)
        toc = time.perf_counter()

        latency.append(toc - tic)

        # Sanity check the output before saving.
        assert len(output) == 3
        assert set(output.keys()) == set(('boxes', 'scores', 'classes'))
        num_objs = output['boxes'].shape[0]
        #print(f'num_objs:{num_objs}')
        assert output['scores'].shape[0] == num_objs
        assert output['classes'].shape[0] == num_objs

        # Save the outputs as numpy files.
        # for k, v in output.items():
        #     npfilename=imagename+'_'+k #k add image name before the key
        #     np.save(os.path.join(output_dir, npfilename), v)
        result_dict[imagename]=output
        
        del data
    # # Save the list of input fields in a text file.
    # with open(os.path.join(output_dir, 'input_fields.txt'), 'w') as f:
    #     f.write('\n'.join(wod_latency_submission.DATA_FIELDS))
    npfilename=os.path.join(output_dir, 'allcameraresult.npy')
    np.save(npfilename, result_dict)
    del result_dict

    # Return the elapsed time of the run_model call.
    latency_np=np.array(latency)
    return np.mean(latency_np)

# class args:
#     nameprefix = "0603dtrn2valall"
#     input_data_dir = "/data/cmpe295-liu/Waymodicts/valdation/"
#     output_dir = "/home/010796032/MyRepo/myoutputs/"+nameprefix+"/"
#     latency_result_file = "/home/010796032/MyRepo/myoutputs/"+nameprefix+".txt"
#     #detectron2 model
#     # model_path = '/home/010796032/MyRepo/Detectron2output/model_0899999.pth'
#     # config_path = ''
#     #mmdetection model
#     model_path = '/home/010796032/MyRepo/Detectron2output/model_0899999.pth'  # model_final.pth'
#     config_path=''

class args:
    nameprefix = "608mm3dkittivalall"
    input_data_dir = "/DATA5T/HPC295Data/Waymodicts/valdation/"
    output_dir = "/Developer/MyRepo/3doutput/"+nameprefix+"/"
    latency_result_file = "/Developer/MyRepo/3doutput/"+nameprefix+".txt"
    
    # model_path = '/Developer/MyRepo/mymodels/detectron2models/model_0899999.pth'
    # config_path = ''

if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input_data_dir', type=str, required=True)
    # parser.add_argument('--output_dir', type=str, required=True)
    # parser.add_argument('--latency_result_file', type=str, required=True)
    # args = parser.parse_args()

    # outputsubmissionfilepath = os.path.join(savepath,args.nameprefix+".bin")#"/home/010796032/MyRepo/WaymoObjectDetection/output/"+nameprefix+".bin"
    #wod_latency_submission.setupmodeldir(args.model_path, args.config_path)

    # Run any user-specified initialization code for their submission.
    wod_latency_submission.initialize_model()

    latencies = []
    contextfileid = 0
    # Iterate through the subdirectories for each frame.
    for context_name in os.listdir(args.input_data_dir):
        context_dir = os.path.join(args.input_data_dir, context_name)
        context_latency = []
        print(f"index: {contextfileid}, context_dir:{context_dir}")
        contextfileid = contextfileid +1
        if not os.path.isdir(context_dir):
            continue
        for timestamp_micros in os.listdir(context_dir):
            timestamp_dir = os.path.join(context_dir, timestamp_micros)
            if not os.path.isdir(timestamp_dir):
                continue

            out_dir = os.path.join(
                args.output_dir, context_name, timestamp_micros)
            os.makedirs(out_dir, exist_ok=True)
            #print('Processing', context_name, timestamp_micros)
            latencyresult = process_example(timestamp_dir, out_dir)# process lidar
            print(f'Processed timestamp_micros: {timestamp_micros}, latency: {latencyresult}')
            latencies.append(latencyresult)
            context_latency.append(latencyresult)
        # converting list to array
        context_latency_np = np.array(context_latency)
        np.save(os.path.join(args.output_dir, context_name, 'latency.npy'), context_latency_np)

    # Save all the latency values in a text file.
    with open(args.latency_result_file, 'w') as latency_file:
        latency_file.write('\n'.join(str(l) for l in latencies))
