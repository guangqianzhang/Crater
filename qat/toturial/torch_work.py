import torch
"""
1、Tensor.register_hook()
2、torch.nn.Module.register_forward_hook()
3、torch.nn.Module.register_backward_hook()
4、torch.nn.Module.register_forward_pre_hook()
"""

"""
Tensor.register_hook:导出张量的梯度
hook函数应该尽可能快地执行，以避免对模型的计算时间造成过多的影响
如果注册了太多的hook函数，会导致额外的内存占用和计算负担。因此，应该仔细考虑何时需要注册hook函数，并在使用后及时删除它们。
"""
def grad_hook(grad):
    # grad *=2
    print(grad)

x=torch.tensor([2.,2.,2.,2.],requires_grad=True)
y= torch.pow(x,2)
z= torch.mean(y)
print(f'z={z}')
hook_handle =x.register_hook(grad_hook)  # 注册一个hook
z.backward()  # 计算z的梯度

print(f'x.grad:{x.grad}')

hook_handle .remove()  # 删除hook

"""
torch.nn.Module.register_forward_hook(): 导出指定模块的输入输出张量，只可以修改输出==导出 修改特征图
"""
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        return x

def hook(module, input, output):
    print(f"out shape: {output.shape}")

model = MyModel()
handle = model.conv2.register_forward_hook(hook)  # 当模型进行前向传递时，hook函数将被调用并打印输出张量的形状

x = torch.randn(1, 3, 224, 224)
output = model(x)
print(f"output shape :{output.shape}")
handle.remove()

"""
torch.nn.Module.register_backward_hook: 修改指定模块的输入张量
"""
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.avgpool(x)
        return x

def hook(module, grad_input, grad_output):
    """grad_input、grad_output 都是元组，返回时也应是元组"""
    print(grad_input[0].shape, grad_output[0].shape)
    gouts=grad_output[0].data.cpu().numpy()
    gin0,gin1=grad_input
    return tuple([gin0,gin1])

model = MyModel()
handle = model.conv2.register_backward_hook(hook)  # 在模型的反向传递中被调用

x = torch.randn(1, 3, 224,224)
output = model(x)
output.sum().backward()

handle.remove()

"""
torch.nn.Module.register_forward_pre_hook(hook): 修改指定张量的输入输出梯度 ，只可修改输入张量梯度
"""
import torch.nn as nn

class MyModel(nn.Module):
    def init(self):
        super(MyModel, self).init()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

def forward(self, x):
    x = self.conv1(x)
    x = self.relu(x)
    x = self.conv2(x)
    x = self.relu(x)
    x = self.avgpool(x)
    return x
    
def hook(module, input):
    print(input[0].shape)

model = MyModel()
handle = model.conv1.register_forward_pre_hook(hook)

x = torch.randn(1, 3, 224, 224)
output = model(x)

handle.remove()
