import torch
from mmdet3d.apis import (inference_mono_3d_detector, init_model,
                          show_result_meshlab)
import mmcv
# model= torch.load('/home/zgq/code/DCMMDet3D/work_dirs/fcos3d_r50_dc/epoch_12.pth')
# model=model['state_dict']
# O_model_name=[]
# for name, module in model.items():
#     O_model_name.append(name)


ptq_model=torch.load('/home/zgq/code/DCMMDet3D/work_dirs/ckpt/det_ptq.pth')
ptq_model=ptq_model.module   ## 加载ptq_model，注意模型的保存方式和推理时的加载方式
config = mmcv.Config.fromfile('configs/fcos3d/fcos3d_r50_dc.py')
epoch=100
import time
t1=time.perf_counter()
for _ in range(epoch):
    result, data = inference_mono_3d_detector(ptq_model, config,'demo/data/nuscenes/4.png', 'demo/data/nuscenes/1.json')
t2=time.perf_counter()
print(f'Avg time: {(t2-t1)/epoch} s')
# show the results
show_result_meshlab(
    data,
    result,
    'demo',
    0.18,
    show=True,
    snapshot=True,
    task='mono-det')


# P_model_name=[]
# for name, module in ptq_model.module.named_children():
#     P_model_name.append(name)

# for orin,ptq in zip(O_model_name,P_model_name):
#     print(f'{orin} - {ptq}')

