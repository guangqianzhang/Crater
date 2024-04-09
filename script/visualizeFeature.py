import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# 加载预训练的ResNet模型
model = models.resnet50(pretrained=True)
model.eval()

# 加载并预处理图像
image_path = r"c:\Users\guangqian zhang\Pictures\Saved Pictures\淄博.jpg"
image = Image.open(image_path).convert("RGB")
preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
input_tensor = preprocess(image)
input_batch = input_tensor.unsqueeze(0)

# 使用模型获取中间层激活
activation_layers = [model.layer1, model.layer2, model.layer3, model.layer4]
activations = []

def hook_fn(module, input, output):
    activations.append(output)

hooks = []
for layer in activation_layers:
    hook = layer.register_forward_hook(hook_fn)
    hooks.append(hook)

# 前向传播
with torch.no_grad():
    model(input_batch)

# 移除注册的hook
for hook in hooks:
    hook.remove()

# 可视化中间层激活
for i, activation in enumerate(activations, 1):
    plt.figure(figsize=(15, 8))
    for j in range(5):  # 可视化前5个通道
        plt.subplot(1, 5, j + 1)
        plt.imshow(activation[0, j].cpu().numpy(), cmap="viridis")
        plt.title(f"Channel {j}")
        plt.axis("off")
    plt.suptitle(f"Layer {i} Activation")
    plt.show()
