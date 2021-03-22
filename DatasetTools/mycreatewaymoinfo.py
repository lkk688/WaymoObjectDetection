import argparse
from os import path as osp
from pathlib import Path

import pandas as pd
import numpy as np
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import mmcv

from myWaymoinfo_utils import get_waymo_image_info, _calculate_num_points_in_gt

def create_trainvaltestsplitfile(dataset_dir, output_dir):
    trainingdir = os.path.join(dataset_dir, 'image_0')
    ImageSetdir = os.path.join(output_dir, 'ImageSets')
    if not os.path.exists(ImageSetdir):
        os.makedirs(ImageSetdir)

    images = os.listdir(trainingdir)
    # totalimages=len([img for img in images])
    # print("Total images:", totalimages)
    dataset = []
    for img in images:
        dataset.append(img[:-4])#remove .png
    print("Total images:", len(dataset))
    df = pd.DataFrame(dataset, columns=['index'], dtype=np.int32)
    X_train, X_val = train_test_split(df, train_size=0.8, test_size=0.2, random_state=42)
    print("Train size:", X_train.shape)
    print("Val size:", X_val.shape)
    write_to_file(os.path.join(ImageSetdir, 'trainval.txt'), df.sort_values('index')['index'])
    write_to_file(os.path.join(ImageSetdir, 'train.txt'), X_train.sort_values('index')['index'])
    write_to_file(os.path.join(ImageSetdir, 'val.txt'), X_val.sort_values('index')['index'])

    # testdir = os.path.join(dataset_dir, 'test', 'image_2')
    # testimages = os.listdir(testdir)
    # # totalimages=len([img for img in images])
    # # print("Total images:", totalimages)
    # testdataset = []
    # for img in testimages:
    #     testdataset.append(img[:-4])#remove .png
    # dftest = pd.DataFrame(testdataset, columns=['index'], dtype=np.int32)
    # print("Test size:", dftest.shape)
    # write_to_file(os.path.join(ImageSetdir, 'test.txt'), dftest.sort_values('index')['index'])


def write_to_file(path, data): 
    file = open(path, 'w') 
    for idx in data: 
        #print(idx)
        file.write(str(idx).zfill(6))
        file.write('\n')

    file.close()
    print('Done in ' + path)

def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]
  
def create_waymotrain_info_file(data_path,
                           pkl_prefix='waymo',
                           save_path=None,
                           relative_path=True,
                           max_sweeps=5):
    """Create info file of waymo dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str | None): Path to save the info file.
        relative_path (bool): Whether to use relative path.
        max_sweeps (int): Max sweeps before the detection frame to be used.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    #val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    #test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
    waymo_infos_train = get_waymo_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        pose=True,
        image_ids=train_img_ids,
        relative_path=relative_path,
        max_sweeps=max_sweeps)
    _calculate_num_points_in_gt(
        data_path,
        waymo_infos_train,
        relative_path,
        num_features=4,#6
        remove_outside=False)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Waymo info train file is saved to {filename}')
    mmcv.dump(waymo_infos_train, filename)

    # waymo_infos_val = get_waymo_image_info(
    #     data_path,
    #     training=True,
    #     velodyne=True,
    #     calib=True,
    #     pose=True,
    #     image_ids=val_img_ids,
    #     relative_path=relative_path,
    #     max_sweeps=max_sweeps)
    # _calculate_num_points_in_gt(
    #     data_path,
    #     waymo_infos_val,
    #     relative_path,
    #     num_features=6,
    #     remove_outside=False)
    # filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    # print(f'Waymo info val file is saved to {filename}')
    # mmcv.dump(waymo_infos_val, filename)

    # filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    # print(f'Waymo info trainval file is saved to {filename}')
    # mmcv.dump(waymo_infos_train + waymo_infos_val, filename)
    # waymo_infos_test = get_waymo_image_info(
    #     data_path,
    #     training=False,
    #     label_info=False,
    #     velodyne=True,
    #     calib=True,
    #     pose=True,
    #     image_ids=test_img_ids,
    #     relative_path=relative_path,
    #     max_sweeps=max_sweeps)
    # filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    # print(f'Waymo info test file is saved to {filename}')
    # mmcv.dump(waymo_infos_test, filename)

def create_waymotrainval_info_file(data_path,
                           pkl_prefix='waymo',
                           save_path=None,
                           relative_path=True,
                           max_sweeps=5):
    """Create info file of waymo dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        data_path (str): Path of the data root.
        pkl_prefix (str): Prefix of the info file to be generated.
        save_path (str | None): Path to save the info file.
        relative_path (bool): Whether to use relative path.
        max_sweeps (int): Max sweeps before the detection frame to be used.
    """
    imageset_folder = Path(data_path) / 'ImageSets'
    train_img_ids = _read_imageset_file(str(imageset_folder / 'train.txt'))
    val_img_ids = _read_imageset_file(str(imageset_folder / 'val.txt'))
    trainval_img_ids = _read_imageset_file(str(imageset_folder / 'trainval.txt'))
    #test_img_ids = _read_imageset_file(str(imageset_folder / 'test.txt'))

    print('Generate info. this may take several minutes.')
    if save_path is None:
        save_path = Path(data_path)
    else:
        save_path = Path(save_path)
        
    waymo_infos_train = get_waymo_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        pose=True,
        image_ids=train_img_ids,
        relative_path=relative_path,
        max_sweeps=max_sweeps)
    _calculate_num_points_in_gt(
        data_path,
        waymo_infos_train,
        relative_path,
        num_features=4,#6
        remove_outside=False)
    filename = save_path / f'{pkl_prefix}_infos_train.pkl'
    print(f'Waymo info train file is saved to {filename}')
    mmcv.dump(waymo_infos_train, filename)

    waymo_infos_val = get_waymo_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        pose=True,
        image_ids=val_img_ids,
        relative_path=relative_path,
        max_sweeps=max_sweeps)
    _calculate_num_points_in_gt(
        data_path,
        waymo_infos_val,
        relative_path,
        num_features=4,#6
        remove_outside=False)
    filename = save_path / f'{pkl_prefix}_infos_val.pkl'
    print(f'Waymo info val file is saved to {filename}')
    mmcv.dump(waymo_infos_val, filename)

    waymo_infos_trainval = get_waymo_image_info(
        data_path,
        training=True,
        velodyne=True,
        calib=True,
        pose=True,
        image_ids=trainval_img_ids,
        relative_path=relative_path,
        max_sweeps=max_sweeps)
    _calculate_num_points_in_gt(
        data_path,
        waymo_infos_trainval,
        relative_path,
        num_features=4,
        remove_outside=False)
    filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    print(f'Waymo info trainval file is saved to {filename}')
    mmcv.dump(waymo_infos_trainval, filename)

    # filename = save_path / f'{pkl_prefix}_infos_trainval.pkl'
    # print(f'Waymo info trainval file is saved to {filename}')
    # mmcv.dump(waymo_infos_train + waymo_infos_val, filename)
    # waymo_infos_test = get_waymo_image_info(
    #     data_path,
    #     training=False,
    #     label_info=False,
    #     velodyne=True,
    #     calib=True,
    #     pose=True,
    #     image_ids=test_img_ids,
    #     relative_path=relative_path,
    #     max_sweeps=max_sweeps)
    # filename = save_path / f'{pkl_prefix}_infos_test.pkl'
    # print(f'Waymo info test file is saved to {filename}')
    # mmcv.dump(waymo_infos_test, filename)

def createwaymo_info(root_path,
                    info_prefix,
                    out_dir,
                    workers,
                    max_sweeps=5):
    """Prepare the info file for waymo dataset.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        out_dir (str): Output directory of the generated info file.
        workers (int): Number of threads to be used.
        max_sweeps (int): Number of input consecutive frames. Default: 5 \
            Here we store pose information of these frames for later use.
    """
    # Generate waymo infos
    #out_dir = osp.join(out_dir, 'kitti_format')
    # Create ImageSets, train test split
    #create_trainvaltestsplitfile(out_dir)
    print("Start to create waymo info .pkl files")
    print("root_path:", root_path)
    create_waymotrain_info_file(root_path, info_prefix, save_path=out_dir, max_sweeps=max_sweeps)

    # kitti.create_waymo_info_file(out_dir, info_prefix, max_sweeps=max_sweeps)
    # create_groundtruth_database(
    #     'WaymoDataset',
    #     out_dir,
    #     info_prefix,
    #     f'{out_dir}/{info_prefix}_infos_train.pkl',
    #     relative_path=False,
    #     with_mask=False)

parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument(
    '--root_path',
    type=str,
    default='/data/cmpe249-f20/WaymoKittitMulti/trainall/',
    help='specify the root path of dataset')
parser.add_argument(
    '--out_dir',
    type=str,
    default='/data/cmpe249-f20/WaymoKittitMulti/trainall/',
    help='out put folder')
parser.add_argument(
        '--createsplitfile_only',
        action='store_true',
        help='create train val split files')
parser.add_argument(
        '--createinfo_only',
        action='store_true',
        help='create info files')
parser.add_argument('--extra-tag', type=str, default='waymo')
parser.add_argument(
    '--workers', type=int, default=2, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    print("Root path:", args.root_path)
    print("out_dir path:", args.out_dir)
    if args.createsplitfile_only:
        create_trainvaltestsplitfile(args.root_path, args.out_dir)
    
    if args.createinfo_only:
        #createwaymo_info(args.root_path, args.extra_tag, args.out_dir, args.workers, max_sweeps=5)
        create_waymotrainval_info_file(args.root_path, args.extra_tag, args.out_dir, args.workers, max_sweeps=5)
    # if args.dataset == 'waymo':
    #     if args.createinfo_only:
    #         else:
    #         waymo_data_prep(
    #             root_path=args.root_path,
    #             info_prefix=args.extra_tag,
    #             version=args.version,
    #             out_dir=args.out_dir,
    #             workers=args.workers,
    #             max_sweeps=args.max_sweeps)