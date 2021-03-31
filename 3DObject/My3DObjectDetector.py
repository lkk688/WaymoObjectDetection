
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

        # build the data pipeline
        self.test_pipeline = deepcopy(config.data.test.pipeline)
        self.test_pipeline = Compose(self.test_pipeline)
        box_type_3d, box_mode_3d = get_box_type(config.data.test.box_type_3d)
        self.data = dict(
            pts_filename='',#pcd,
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
            # strip prefix of state_dict
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
    
    def detect(self, pcd):
        self.data['pts_filename']=pcd

        data = self.test_pipeline(self.data)
        data = collate([data], samples_per_gpu=1)
        if next(self.model.parameters()).is_cuda:
            # scatter to specified GPU
            data = scatter(data, [self.device.index])[0]
        # forward the model
        with torch.no_grad():
            result = self.model(return_loss=False, rescale=True, **data)
        #return result, data
        return result
