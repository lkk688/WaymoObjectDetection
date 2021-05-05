import torch
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name())
print(torch.__version__)

# Check Pytorch installation
import torchvision
print(torchvision.__version__, torch.cuda.is_available())

# Check MMDetection installation
import mmdet
print(mmdet.__version__)

# Check mmcv installation
from mmcv.ops import get_compiling_cuda_version, get_compiler_version
print(get_compiling_cuda_version())
print(get_compiler_version())

import argparse
import copy
import os
import os.path as osp
import time
import warnings

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.runner import get_dist_info, init_dist
from mmcv.utils import get_git_hash

from mmdet import __version__
from mmdet.apis import set_random_seed, train_detector
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
import numpy as np

class mmargs:
    #Basemmdetection='/Developer/3DObject/mmdetection/'
    MyBasemmdetection='/Developer/3DObject/mymmdetection/'
    config = MyBasemmdetection+'configs/faster_rcnn/myfaster_rcnn_x101_64x4d_fpn_1x_coco.py'
    # Setup a checkpoint file to load
    checkpoint = MyBasemmdetection+'checkpoints/faster_rcnn_x101_64x4d_fpn_1x_coco_20200204-833ee192.pth'
    workdir = MyBasemmdetection+"waymococo_fasterrcnntrain"
    resumefrom = None #basefolder+ 'myresults/epoch_120.pth'
    novalidate = False
    gpus = 1 
    gpuids = None
    seed =None
    deterministic=True
    classes=('vehicle', 'pedestrian', 'sign', 'cyclist')#('person', 'bicycle', 'car')
    data_root = '/DATA5T/Dataset/WaymoCOCO/'

class mmargsfasterr101:
    #Basemmdetection='/Developer/3DObject/mmdetection/'
    MyBasemmdetection='/Developer/3DObject/mymmdetection/'
    config = MyBasemmdetection+'configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py'
    # Setup a checkpoint file to load
    checkpoint = MyBasemmdetection+'checkpoints/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'
    workdir = MyBasemmdetection+"waymococo_fasterrcnnr101train"
    resumefrom = None #basefolder+ 'myresults/epoch_120.pth'
    novalidate = False
    gpus = 1 
    gpuids = None
    seed =None
    deterministic=True
    classes=('vehicle', 'pedestrian', 'sign', 'cyclist')#('person', 'bicycle', 'car')
    data_root = '/DATA5T/Dataset/WaymoCOCO/'

def maintrain(args):
    cfg = Config.fromfile(args.config)

    # New add to setup the dataset, no need to change the configuration file
    cfg.dataset_type = 'CocoDataset'
    cfg.data.test.type = 'CocoDataset'
    cfg.data.test.data_root = args.data_root 
    cfg.data.test.ann_file = args.data_root + 'annotations_val20new.json' #'annotations_valallnew.json'
    cfg.data.test.img_prefix =  ''

    cfg.data.train.type = 'CocoDataset'
    cfg.data.train.data_root = args.data_root
    cfg.data.train.ann_file = args.data_root + 'annotations_train20new.json' #'annotations_trainallnew.json'
    cfg.data.train.img_prefix = ''

    cfg.data.val.type = 'CocoDataset'
    cfg.data.val.data_root = args.data_root
    cfg.data.val.ann_file = args.data_root + 'annotations_val20new.json' #'annotations_valallnew.json'
    cfg.data.val.img_prefix = ''

    #batch size=2, workers=0, eta: 1 day, 5:56:54,  memory: 5684
    cfg.data.samples_per_gpu = 4 #batch size
    cfg.data.workers_per_gpu = 4
    #eta: 1 day, 6:17:04, memory: 10234

    # modify num classes of the model in box head
    cfg.model.roi_head.bbox_head.num_classes = len(args.classes)# 4

    # import modules from string list.
    if cfg.get('custom_imports', None):#not used
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg['custom_imports'])
    # set cudnn_benchmark, benchmark mode is good whenever your input sizes for your network do not vary. This way, cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). This usually leads to faster runtime.
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.workdir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.workdir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])
    # create work_dir
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))

    # dump config
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    # init the logger before other steps
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

    cfg.load_from = args.checkpoint
    if args.resumefrom is not None:
        cfg.resume_from = args.resumefrom
    if args.gpuids is not None:
        cfg.gpu_ids = args.gpuids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)

    # init the meta dict to record some important information such as
    # environment info and seed, which will be logged
    meta = dict()
    # log env info
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' +
                dash_line)
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    # log some basic info
    distributed = False
    logger.info(f'Distributed training: {distributed}')
    logger.info(f'Config:\n{cfg.pretty_text}')

    # set random seeds
    if args.seed is not None: #not used
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))
    if cfg.checkpoint_config is not None:
        # save mmdet version, config file content and class names in
        # checkpoints as meta data
        cfg.checkpoint_config.meta = dict(
            mmdet_version=__version__ + get_git_hash()[:7],
            CLASSES=datasets[0].CLASSES)
    # add an attribute for visualization convenience
    model.CLASSES = args.classes #datasets[0].CLASSES
    train_detector(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.novalidate),
        timestamp=timestamp,
        meta=meta)


if __name__ == "__main__":
    maintrain(mmargsfasterr101)#(mmargs)
    