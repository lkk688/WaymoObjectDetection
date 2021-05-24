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

from PIL import Image
classes=('vehicle', 'pedestrian', 'sign', 'cyclist')
def mymm2d_init_detector(config, checkpoint=None, device='cuda:0'):
    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    config.model.train_cfg = None
    config.model.roi_head.bbox_head.num_classes = len(classes)
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

def testinference(config, checkpoint):
    #Basemmdetection='/Developer/3DObject/mmdetection/'
    #config = Basemmdetection+'configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py'
    # Setup a checkpoint file to load
    #checkpoint = Basemmdetection+'checkpoints/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'

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

#REF: https://github.com/open-mmlab/mmdetection/blob/master/mmdet/apis/inference.py
from mmdet.datasets import replace_ImageToTensor
from mmdet.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
def myinferencedetector(model, img):
    image = Image.open(img)
    # summarize some details about the image
    print(image.format)
    print(image.size)
    print(image.mode)
    # convert image to numpy array
    image_np = np.asarray(image)
    print(type(image_np))
    # summarize shape
    print(image_np.shape) #(1280, 1920, 3)

    datas = []

    cfg = model.cfg
    device = next(model.parameters()).device  # model device
    cfg = cfg.copy()
    # set loading pipeline type
    cfg.data.test.pipeline[0].type = 'LoadImageFromWebcam'
    cfg.data.test.pipeline = replace_ImageToTensor(cfg.data.test.pipeline)
    test_pipeline = Compose(cfg.data.test.pipeline)

    if isinstance(image_np, np.ndarray):
        # directly add img
        data = dict(img=image_np)
    # build the data pipeline
    data = test_pipeline(data)
    datas.append(data)

    data = collate(datas, samples_per_gpu=1)
    # just get the actual data from DataContainer
    data['img_metas'] = [img_metas.data[0] for img_metas in data['img_metas']]
    data['img'] = [img.data[0] for img in data['img']]

    data = scatter(data, [device])[0]
    # forward the model
    with torch.no_grad():
        results = model(return_loss=False, rescale=True, **data)
    return results[0]


def testinferencedetector(config, checkpoint):
    #Basemmdetection='/Developer/3DObject/mmdetection/'
    #config = Basemmdetection+'configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py'
    # Setup a checkpoint file to load
    #checkpoint = Basemmdetection+'checkpoints/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'

    # initialize the detector
    # model1 = init_detector(config, checkpoint, device='cuda:0')
    model1 = mymm2d_init_detector(config, checkpoint, device='cuda:0')

    img = '/DATA5T/Dataset/WaymoKitti/4c_train5678/training/image_0/000010.png'
    #result = inference_detector(model1, img) #result (tuple[list] or list): The detection result, can be either (bbox, segm) or just bbox.
    result = myinferencedetector(model1, img)
    show_result_pyplot(model1, img, result, score_thr=0.3)

    bbox_result=result
    bboxes = np.vstack(bbox_result)
    labels = [
        np.full(bbox.shape[0], i, dtype=np.int32)
        for i, bbox in enumerate(bbox_result)
    ]
    labels = np.concatenate(labels)
    print(bboxes)#5,1, last column is the confidence score
    #xmin, ymin, xmax, ymax in pixels
    print(type(bboxes))
    print(type(labels))

    num_box=len(bboxes)
    newboxes = np.zeros((num_box,4), dtype=float)
    newboxes[:, 0] = (bboxes[:, 0] + bboxes[:, 2]) / 2.0 #center_x
    newboxes[:, 1] = (bboxes[:, 1] + bboxes[:, 3])  / 2.0 #center_y
    newboxes[:, 2] = (bboxes[:, 2] - bboxes[:, 0]) #width
    newboxes[:, 3] = (bboxes[:, 3] - bboxes[:, 1]) # height
    # boxes[:, 0] = (pred_boxes[:, 3] + pred_boxes[:, 1]) * im_width / 2.0
    # boxes[:, 1] = (pred_boxes[:, 2] + pred_boxes[:, 0]) * im_height / 2.0
    # boxes[:, 2] = (pred_boxes[:, 3] - pred_boxes[:, 1]) * im_width
    # boxes[:, 3] = (pred_boxes[:, 2] - pred_boxes[:, 0]) * im_height
    pred_score = np.zeros((num_box,), dtype=float)
    pred_score = bboxes[:,4]
    return {
        'boxes': newboxes,
        'scores': pred_score,
        'classes': labels,
    }

if __name__ == "__main__":
    Basemmdetection='/Developer/3DObject/mmdetection/'
    config = Basemmdetection+'configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py'
    # Setup a checkpoint file to load
    checkpoint = Basemmdetection+'checkpoints/faster_rcnn_r101_fpn_2x_coco_bbox_mAP-0.398_20200504_210455-1d2dac9c.pth'
    #testinference(config, checkpoint)

    Basemmdetection='/Developer/3DObject/mmdetection/'
    config = Basemmdetection+'configs/faster_rcnn/faster_rcnn_r101_fpn_2x_coco.py'
    checkpoint = '/Developer/3DObject/mymmdetection/waymococo_fasterrcnnr101train/epoch_60.pth'
    testinferencedetector(config, checkpoint)