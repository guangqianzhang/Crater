"""
QuantConv1d、QuantConv2d、Quant
Conv3d、QuantConvTranspose1d、quantConvTranspose 2d、QuantiConvTransbose
3d、QuantiLinear、QuantAvgPool1d、QuantAvg Pool2d、Quant AvgPool
3d、QuantMaxPool1d、QuantMax Pool2D、Quant MaxPool3d
"""

import torch
from pytorch_quantization import tensor_quant
#国定种子12345并生成随机输入X为:
# tensor([0.9817,8796,0,9921, 0,4611, 0,0832, 0,1784,0,3674, 0,5676,0,3376,0,2119])
torch.manual_seed(12345)
x = torch.rand(10)
print(f"x:            {x}")
#伪量化张量 x:
#tensor([0.9843, 0.8828,0.9921, 0.4609 .0859, 0,1797, 0.3672, 0.5703, 0.3359, 0.2109])
fake_quant_x = tensor_quant.fake_tensor_quant(x, x.abs().max())
print(f"fake_quant_x:{fake_quant_x}")
# 量化张量x，scale=128.8857:
# tensor([126.，113.， 127.，59., 11., 23., 47., 73.，43.,27.J)
quant_x, scale = tensor_quant.tensor_quant(x, x.abs().max())
print(f"quant_x:{quant_x},scale:{scale}")


from torch import nn
from pytorch_quantization import tensor_quant
import pytorch_quantization.nn as quant_nn

in_features=32
out_features=10
in_channels=3
out_channels=32
kernel_size=3

# PyTorch模型
fc1 = nn.Linear(in_features, out_features, bias=True)
conv1 = nn.Conv2d(in_channels, out_channels, kernel_size)

# 量化版本的模型
quant_fc1 = quant_nn.Linear(
    in_features, out_features, bias=True,
    quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
    quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_LINEAR_WEIGHT_PER_ROW
)

quant_conv1 = quant_nn.Conv2d(
    in_channels, out_channels, kernel_size,
    quant_desc_input=tensor_quant.QUANT_DESC_8BIT_PER_TENSOR,
    quant_desc_weight=tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL
)
