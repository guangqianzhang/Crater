
import torch

from tqdm import tqdm
from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib






def evaluate_accuracy(model, dataloader, device,logger):
    correct = 0
    total = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for _,data in enumerate(tqdm(dataloader)):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        logger.info('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
    return correct / total

def fine_tune(model, dataloader, device,logger):
    # 微调模型
    logger.info('Start Fine-tuning..')
    # Fine-tuning
    model.eval()
    for i, (data, target) in enumerate(tqdm(dataloader)):
        data=data.to(device)
        model(data)


def calibrate_model(model : torch.nn.Module, dataloader, device,  num_batch=1):
    """
    对模型进行校准，并计算TensorQuantizer模块中校准器的最大值
    
    Args:
        model (torch.nn.Module): 待校准的模型
        dataloader (DataLoader): 数据加载器，用于提供校准数据
        device (str or torch.device): 设备类型，指定模型和数据运行在哪个设备上
        batch_processor_callback (Callable, optional): 批处理回调函数，用于自定义处理每个batch的函数，默认为None
        num_batch (int, optional): 用于收集统计信息的batch数量，默认为1
    
    Returns:
        None
    """

    def compute_amax(model, **kwargs):
        # 遍历模型的所有模块
        for name, module in model.named_modules():
            # 判断模块是否为 TensorQuantizer 类型
            if isinstance(module, quant_nn.TensorQuantizer):
                # 判断模块是否包含校准器
                if module._calibrator is not None:
                    # 判断校准器是否为 MaxCalibrator 类型
                    if isinstance(module._calibrator, calib.MaxCalibrator):
                        # 加载校准器计算的最大值
                        module.load_calib_amax(strict=False)
                    else:
                        # 加载校准器计算的最大值，并传递额外的参数
                        module.load_calib_amax(strict=False, **kwargs)
                    # 将最大值转移到指定设备上
                    module._amax = module._amax.to(device)
        
    def collect_stats(model, data_loader, device, num_batch=200):
        """Feed data to the network and collect statistics"""
        # Enable calibrators
        model.eval()
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.disable_quant()
                    module.enable_calib()
                else:
                    module.disable()

        iter_count = 0 
        for data in tqdm(data_loader, total=num_batch, desc="Collect stats for calibrating"):
            with torch.no_grad():
                result = model(return_loss=False, rescale=True, **data)
            iter_count += 1
            if iter_count >num_batch:  # 指定收集的batch数量
                break

        # Disable calibrators
        for name, module in model.named_modules():
            if isinstance(module, quant_nn.TensorQuantizer):
                if module._calibrator is not None:
                    module.enable_quant()
                    module.disable_calib()
                else:
                    module.enable()

    collect_stats(model, dataloader, device, num_batch=num_batch)
    compute_amax(model, method="mse")  # 均方误差


