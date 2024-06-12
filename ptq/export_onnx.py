from functools import partial
import onnxruntime
import torch
from mmdet3d.apis import init_model
import mmcv
import os.path as osp
from mmdet3d.core.bbox import get_box_type
from mmdet3d.datasets.pipelines import Compose
from copy import deepcopy
from mmdet3d.core import (Box3DMode)
from mmcv.parallel import collate, scatter
import numpy as np
from mmdeploy.core import RewriterContext, patch_model
config = 'configs/fcos3d/fcos3d_r50_dc.py'
checkpoint = 'work_dirs/fcos3d_r50_dc/epoch_12.pth'
ann_file='demo/data/nuscenes/1.json'
image = 'demo/data/nuscenes/4.png'
onnxfile='fcos3d_r50_dc.onnx'
def build_model(config, checkpoint,device):
    # cfg = mmcv.Config.fromfile(config)
    model = init_model(config, checkpoint, device=device)
    device = next(model.parameters()).device  # model device
    print(f'device on {device}')
    return model.eval()

def perpare_data(cfg, image=None,ann_file=None):

    if image is None:
        image = torch.randn(1, 3, 1216, 1200)
    if ann_file is None:
        print('there are need a json file')
    test_pipeline = deepcopy(cfg.data.test.pipeline)
    test_pipeline = Compose(test_pipeline)
    box_type_3d, box_mode_3d = get_box_type(cfg.data.test.box_type_3d)
    # get data info containing calib
    data_infos = mmcv.load(ann_file)
    # find the info corresponding to this image
    for x in data_infos['images']:
        if osp.basename(x['file_name']) != osp.basename(image):
            continue
        img_info = x
        break
    data = dict(
        img_prefix=osp.dirname(image),
        img_info=dict(filename=osp.basename(image)),
        box_type_3d=box_type_3d,
        box_mode_3d=box_mode_3d,
        img_fields=[],
        bbox3d_fields=[],
        pts_mask_fields=[],
        pts_seg_fields=[],
        bbox_fields=[],
        mask_fields=[],
        seg_fields=[])

    # camera points to image conversion
    if box_mode_3d == Box3DMode.CAM:
        data['img_info'].update(
            dict(cam_intrinsic=img_info['cam_intrinsic']))

    data = test_pipeline(data)

    data = collate([data], samples_per_gpu=1)
    # data['img_metas'] = [
    #     img_metas.data[0] for img_metas in data['img_metas']
    # ]
    # data['img'] = [img.data[0] for img in data['img']]
    # data['cam2img'] = [torch.tensor(data['img_metas'][0][0]['cam2img'])]
    # data['cam2img_inverse'] = [torch.inverse(data['cam2img'][0])]
    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [0])[0]
    else:
        # this is a workaround to avoid the bug of MMDataParallel
        data['img_metas'] = data['img_metas'][0].data
        data['img'] = data['img'][0].data
        
    # if device != 'cpu':
    #     # scatter to specified GPU
    #     data = scatter(data, [device])[0]
    cam2img = torch.tensor(data['img_metas'][0][0]['cam2img'])
    cam2img_inverse=torch.inverse(cam2img)
    return data.pop('img'),data
    # return data
####################################################################################
device='cuda:0'
model= build_model(config, checkpoint,device)
ptq_model=torch.load('/home/zgq/code/DCMMDet3D/work_dirs/ckpt/det_ptq.pth')
ptq_model=ptq_model.module   ## 加载ptq_model，注意模型的保存方式和推理时的加载方式


data, model_inputs= perpare_data(cfg=model.cfg, image=image,ann_file=ann_file)
# data = perpare_data(cfg=model.cfg, image=image,ann_file=ann_file)
data=torch.randn(1, 3, 1216, 1200)

model_inputs=dict
inputs_name=['img','cam2img','cam2img_inverse'] # 1x3x1216x1200 3x3 3x3
outputs_name=['bboxes', 'scores', 'labels','dir_scores','attrs'] # 

# if not isinstance(model_inputs, torch.Tensor) and len(model_inputs) == 1:
#     model_inputs = [model_inputs[0].cuda()]  # 因为base文件中使用img[0] iamg_metas[0]传入


# model=build_model(config=config,checkpoint=checkpoint,device='cpu')
# patched_model = patch_model(model, cfg=deploy_cfg, backend='backend', ir=IR.ONNX)
# patched_model=model

with torch.no_grad():
    model_forward = ptq_model.forward
    ptq_model.forward = partial(ptq_model.forward, **model_inputs)
    # ptq_model(return_loss=False,**model_inputs)  # 推理调用前向传播方法

    torch.onnx.export(
        ptq_model,
        model_inputs, 
        onnxfile,
        opset_version=12,
        input_names=inputs_name,
        output_names=outputs_name)

    # 恢复原始的前向传播方法
    # model.forward = model_forward
