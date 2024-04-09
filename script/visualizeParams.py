import torch
import torch.nn as nn

def analyze_model(model):
    # 打印模型结构
    print(model)

    # 获取所有层
    layers = list(model.children())

    # 初始化参数计数
    total_params = 0
    trainable_params = 0
    non_trainable_params = 0

    # 遍历每一层，获取形状和参数信息
    for layer in layers:
        print(f"\nLayer Name: {layer.__class__.__name__}")
        print(f"Input Shape: {layer.in_features  if hasattr(layer,'in_features') else 'N/A'}")
        print(f"Output Shape: {layer.out_features if hasattr(layer, 'out_features') else 'N/A'}")

        # 获取层的参数
        params = list(layer.parameters())
        layer_params = sum(p.numel() for p in params)  # 计算参数数量 权重和偏置
        total_params += layer_params

        # 区分可训练和不可训练参数
        trainable_params += sum(p.numel() for p in params if p.requires_grad)
        # non_trainable_params += layer_params - trainable_params

    # 打印总体参数信息
    print("\nTotal Parameters: {:,}".format(total_params))
    print("Trainable Parameters: {:,}".format(trainable_params))
    print("Non-Trainable Parameters: {:,}".format(total_params - trainable_params))

# 示例模型
class SampleModel(nn.Module):
    def __init__(self):
        super(SampleModel, self).__init__()
        self.fc1 = nn.Linear(100, 50)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(50, 10)

        self.fc2.weight.requires_grad = False
        self.fc2.bias.requires_grad = False

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 创建示例模型
model = SampleModel()

# 分析模型
analyze_model(model)
