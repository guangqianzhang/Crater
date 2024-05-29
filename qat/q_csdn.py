# QAT follows the same steps as PTQ, with the exception of the training loop before you actually convert the model to its quantized version
# QAT遵循与PTQ相同的步骤，除了在实际将模型转换为量化版本之前进行训练循环
''''''
'''量化感知训练步骤：
step1.搭建模型
step2.融合（可选步骤）
step3.插入stubs（1和3可合在一起）
step4.准备（主要是选择架构）
step5.训练
step6.模型转换
'''
import torch
from torch import nn

backend = "fbgemm"  # running on a x86 CPU. Use "qnnpack" if running on ARM.

'''step1.搭建模型build model'''
m = nn.Sequential(
     nn.Conv2d(2,64,8),
     nn.ReLU(),
     nn.Conv2d(64, 128, 8),
     nn.ReLU(),
)

"""step2.融合Fuse（可选步骤）"""
torch.quantization.fuse_modules(m, ['0','1'], inplace=True) # fuse first Conv-ReLU pair
torch.quantization.fuse_modules(m, ['2','3'], inplace=True) # fuse second Conv-ReLU pair

"""step3.插入stubs于模型，Insert stubs"""
m = nn.Sequential(torch.quantization.QuantStub(),
                  *m,
                  torch.quantization.DeQuantStub())

"""step4.准备Prepare"""
m.train()
m.qconfig = torch.quantization.get_default_qconfig(backend)
torch.quantization.prepare_qat(m, inplace=True)

"""step5.训练Training Loop"""
n_epochs = 10
opt = torch.optim.SGD(m.parameters(), lr=0.1)
loss_fn = lambda out, tgt: torch.pow(tgt-out, 2).mean()
for epoch in range(n_epochs):
  x = torch.rand(10,2,24,24)
  out = m(x)
  loss = loss_fn(out, torch.rand_like(out))
  opt.zero_grad()
  loss.backward()
  opt.step()
  print(loss)

"""step6.模型转换Convert"""
m.eval()
torch.quantization.convert(m, inplace=True)
