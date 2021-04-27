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

from mmcv import Config
from mmdet.apis import inference_detector, init_detector, show_result_pyplot
from mmdet.models import build_detector
from mmcv.runner import load_checkpoint
from mmdet.core import get_classes
import mmcv
import numpy as np

def mymm2d_init_detector(config, checkpoint=None, device='cuda:0'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    config.model.train_cfg = None
    model = build_detector(config.model, test_cfg=config.get('test_cfg'))
    if checkpoint is not None:
        map_loc = 'cpu' if device == 'cpu' else None
        checkpoint = load_checkpoint(model, checkpoint, map_location=map_loc)
        if 'CLASSES' in checkpoint.get('meta', {}):
            model.CLASSES = checkpoint['meta']['CLASSES']
        else:
            model.CLASSES = get_classes('coco')
    model.cfg = config  # save the config in the model for convenience
    model.to(device)
    model.eval()
    return model

if __name__ == "__main__":
    Basemmdetection='/Developer/3DObject/mmdetection/'
    config = Basemmdetection+'configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py'
    # Setup a checkpoint file to load
    checkpoint = Basemmdetection+'checkpoints/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'

    # initialize the detector
    # model1 = init_detector(config, checkpoint, device='cuda:0')
    model1 = mymm2d_init_detector(config, checkpoint, device='cuda:0')

    img = '/DATA5T/Dataset/WaymoKitti/4c_train5678/training/image_0/000000.png'
    result = inference_detector(model1, img) #result (tuple[list] or list): The detection result, can be either (bbox, segm) or just bbox.
    show_result_pyplot(model1, img, result, score_thr=0.3)

    bbox_result=result
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    print(bboxes)
    print(type(bboxes))
    print(type(labels))