
import os
import numpy as np
from mmdet3d.apis import init_detector, inference_detector, show_result_meshlab
from mmdet3d.models import build_detector
from mmdet3d.core import Box3DMode
import mmcv
from mmcv.parallel import collate, scatter
import torch

from copy import deepcopy
from mmdet3d.datasets.pipelines import Compose
from mmdet3d.core.bbox import get_box_type

def convert_SyncBN(config):
    """Convert config's naiveSyncBN to BN.

    Args:
         config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
    """
    if isinstance(config, dict):
        for item in config:
            if item == 'norm_cfg':
                config[item]['type'] = config[item]['type']. \
                                    replace('naiveSyncBN', 'BN')
            else:
                convert_SyncBN(config[item])

class MyMM3DObjectDetector(object):
    def __init__(self, args):
        self.args = args
        self.use_cuda = self.args.use_cuda#True
        if self.args.use_cuda is True:
            self.device='cuda:0'
        else:
            self.device='cpu'

        config = mmcv.Config.fromfile(self.args.configfile)
        config.model.pretrained = None
        convert_SyncBN(config.model)
        config.model.train_cfg = None

        self.model = build_detector(config.model, test_cfg=config.get('test_cfg'))
        if self.args.checkpoint is not None:
            checkpoint = torch.load(self.args.checkpoint)
            # OrderedDict is a subclass of dict
            if not isinstance(checkpoint, dict):
                raise RuntimeError(
                    f'No state_dict found in checkpoint file')
            # get state_dict from checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            # strip prefix of state_dict, not used here
            if list(state_dict.keys())[0].startswith('module.'):
                state_dict = {k[7:]: v for k, v in state_dict.items()}
            # load state_dict
            self.model.load_state_dict(state_dict)
            #load_state_dict(model, state_dict, strict, logger)
            
            #checkpoint = load_checkpoint(model, self.args.checkpoint)
            if 'CLASSES' in checkpoint['meta']:
                self.model.CLASSES = checkpoint['meta']['CLASSES']
            else:
                self.model.CLASSES = config.class_names
        self.model.cfg = config  # save the config in the model for convenience
        self.model.to(self.device)
        self.model.eval()
    
    def datapipelinefromlidarfile(self, config, pts_filename):
        # build the data pipeline
        self.test_pipeline = deepcopy(config.data.test.pipeline)
        self.test_pipeline = Compose(self.test_pipeline)
        box_type_3d, box_mode_3d = get_box_type(config.data.test.box_type_3d)
        data = dict(
            pts_filename=pts_filename,#pcd,
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d,
            sweeps=[],
            # set timestamp = 0
            timestamp=[0],
            img_fields=[],
            bbox3d_fields=[],
            pts_mask_fields=[],
            pts_seg_fields=[],
            bbox_fields=[],
            mask_fields=[],
            seg_fields=[])
        return data
    
    def datapipelinefrompcd(self, config, pcd):
        # build the data pipeline
        point_cloud_range = [0, -39.68, -3, 69.12, 39.68, 1]
        class_names = ['Pedestrian', 'Cyclist', 'Car']
        my_pipeline = [
            #dict(type='LoadPointsFromFile', coord_type='LIDAR', load_dim=4, use_dim=4),
            dict(
                type='MultiScaleFlipAug3D',
                img_scale=(1333, 800),
                pts_scale_ratio=1,
                flip=False,
                transforms=[
                    dict(
                        type='GlobalRotScaleTrans',
                        rot_range=[0, 0],
                        scale_ratio_range=[1., 1.],
                        translation_std=[0, 0, 0]),
                    dict(type='RandomFlip3D'),
                    dict(
                        type='PointsRangeFilter', point_cloud_range=point_cloud_range),
                    dict(
                        type='DefaultFormatBundle3D',
                        class_names=class_names,
                        with_label=False),
                    dict(type='Collect3D', keys=['points'])
                ])
        ]
        self.test_pipeline = my_pipeline#deepcopy(config.data.test.pipeline)
        self.test_pipeline = Compose(self.test_pipeline)
        box_type_3d, box_mode_3d = get_box_type(config.data.test.box_type_3d)
        data = dict(
            pts_filename=[],#pcd,
            box_type_3d=box_type_3d,
            box_mode_3d=box_mode_3d,
            sweeps=[],
            # set timestamp = 0
            timestamp=[0],
            img_fields=[],
            bbox3d_fields=[],
            pts_mask_fields=[],
            pts_seg_fields=[],
            bbox_fields=[],
            mask_fields=[],
            seg_fields=[])
        return data

    def datafrompcd(self, config, pcd):
        box_type_3d, box_mode_3d = get_box_type(config.data.test.box_type_3d)
        datadict={}
        datadict['points']=pcd
        datadict['img_metas']=dict(flip=False, pcd_horizontal_flip=False, pcd_vertical_flip=False,
            box_mode_3d=box_mode_3d, box_type_3d=box_type_3d,
            pcd_trans=[0., 0., 0.], pcd_scale_factor=1.0, pts_filename='', transformation_3d_flow=['R', 'S','T'])
        data=[datadict]
        return data

    def detectfile(self, pcdfiles):
        #self.data['pts_filename']=pcd
        data = self.datapipelinefromlidarfile(self.model.cfg, pcdfiles)

        device = next(self.model.parameters()).device  # model device
 
        data = self.test_pipeline(data)
        data = collate([data], samples_per_gpu=1)
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device.index])[0]
        # forward the model
        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)
        #return result, data
        boxes_3d = result[0]['boxes_3d'].tensor.numpy()
        scores_3d = result[0]['scores_3d'].numpy()
        labels_3d = result[0]['labels_3d'].numpy()
        return {'boxes': boxes_3d,
            'scores': scores_3d,
            'classes': labels_3d}
    
    def detect(self, pcd):
        #self.data['pts_filename']=pcd
        #data = self.datapipelinefromlidarfile(self.model.cfg, pcd)
        data =self.datafrompcd(self.model.cfg, pcd)

        device = next(self.model.parameters()).device  # model device
 
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [device.index])[0]
        # forward the model
        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)
        #return result, data
        boxes_3d = result[0]['boxes_3d'].tensor.numpy()
        scores_3d = result[0]['scores_3d'].numpy()
        labels_3d = result[0]['labels_3d'].numpy()
        return {'boxes': boxes_3d,
            'scores': scores_3d,
            'classes': labels_3d}
