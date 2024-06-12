import numpy as np
import torch
import torch.nn as nn
import mmcv
from mmcv.cnn.resnet import ResNet
from mmdet.models.necks import FPN
from mmdet3d.models.dense_heads import fcos_mono3d_head

from head import bbox_head
from resnet import ResNet50, BasicBlock




class FCOS3D(torch.nn.Module):
    def __init__(self, config):
        super(FCOS3D, self).__init__()
        self.config = config
        # convert_SyncBN(self.config.model)

        self.backbone_name= self.config.model['backbone'].pop('type')
        self.neck_name= self.config.model['neck'].pop('type')
        self.head_name= self.config.model['bbox_head'].pop('type')
          
        self.backbone = ResNet(**self.config.model['backbone'])
        self.neck = FPN(**self.config.model['neck'])
        head_config=self.config.model['bbox_head']
        # print(head_config)
        hhh=dict( num_classes=1,in_channels=128)
        self.head = bbox_head(**head_config)
        print("ok")

    def _init_head(self, head):
        return head
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.neck(x)
        return self.head(x)



config= mmcv.Config.fromfile('/home/zgq/code/DCMMDet3D/work_dirs/fcos3d_r50_dc/fcos3d_r50_dc.py')
model= FCOS3D(config)
model.eval()
inputdata= torch.tensor(np.random.randn(1, 3, 1216, 1200),dtype=torch.float32)
outputdata=model(inputdata)
print(len(outputdata))

inputs_name=['img','cam2img','cam2img_inverse'] # 1x3x1216x1200 3x3 3x3
outputs_name=['bboxes', 'scores', 'labels','dir_scores','attrs'] # 
torch.onnx.export(
    model,
    inputdata,
    "/home/zgq/code/DCMMDet3D/work_dirs/fcos3.onnx",
    export_params=True,
    opset_version=12,
    input_names=inputs_name,
    output_names=outputs_name) 

                     # torch.onnx.export(
    #     model,
    #     model_inputs, 
    #     onnxfile,
    #     opset_version=12,
    #     input_names=inputs_name,
    #     output_names=outputs_name) 