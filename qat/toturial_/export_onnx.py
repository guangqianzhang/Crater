"""TensorRT将获取该图，并在int8中以最优化的方式执行该图。
首先将TensorQuantizer的静态成员设置为使用Pytorch自己的伪量化函数： quant_nn.TensorQuantizer.use_fb_fake_quant = True
"""
import torch
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import quant_modules
import torchvision.models as models

# 开启 FB 伪量化
quant_nn.TensorQuantizer.use_fb_fake_quant = True

# 初始化量化模块
quant_modules.initialize()

# 加载量化后的 ResNet50 模型
model = models.resnet50()
state_dict = torch.load("quant_resnet50-entropy-1024.pth", map_location="cpu")
model.load_state_dict(state_dict)
model.cuda()

# 创建虚拟输入
dummy_input = torch.randn(128, 3, 224, 224, device='cuda')

# 导出量化后的 ONNX 模型
input_names = ["actual_input_1"]
output_names = ["output1"]
torch.onnx.export(
    model, dummy_input, "quant_resnet50.onnx",
    verbose=True, opset_version=13, input_names=input_names, output_names=output_names,
    enable_onnx_checker=False  # ONNX 检查器可能会检测到不支持的操作，从而导致导出失败
)
