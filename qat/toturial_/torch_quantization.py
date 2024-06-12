import argparse
import os
from tqdm import tqdm
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


from utils.logger import set_logger
from models.mymodel import QuantModel,MyModel
from torch.quantization.observer import MovingAverageMinMaxObserver, MovingAveragePerChannelMinMaxObserver,HistogramObserver

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import calib
x86=True
def  parse_opt():
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', type=str,default=r'E:\workdir\data')
    parser.add_argument('--workers', type=int,default=8)
    parser.add_argument('--weights', type=str,default=r'E:\workdir\Crater\qat\toturial\weight\model.pth')
    parser.add_argument('--quant-type', type=str,default='int8')
    parser.add_argument('--batch_size', type=int,default=4)
    parser.add_argument('--out-dir','-o', type=str,default=r'E:\workdir\Crater\qat\toturial\weights')
    parser.add_argument('--calib-method', type=str,default='MinMaxObserver')
    parser.add_argument('--evaluation', action='store_true',default=True)

    opt= parser.parse_args()
    return opt

def setConfig(calib_method,per_channel_quantization=False):
    # if per_channel_quantization:
    #     calib_method = MovingAveragePerChannelMinMaxObserver.with_args(qscheme=torch.per_channel_affine)
    # else:
    #     calib_method = MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_affine)

    if calib_method == 'MinMaxObserver':
        my_config=torch.quantization.QConfig(
            activation=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_affine),
            weight=MovingAveragePerChannelMinMaxObserver.with_args(qscheme=torch.qint8)
        )
    elif calib_method == 'HistogramObserver':
            my_config=torch.quantization.QConfig(
            activation=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_affine),
            weight=HistogramObserver.with_args(qscheme=torch.qint8)
        )

    print("set config:{}".format(my_config))


def initialize_calib_method(myconfig:torch.quantization.QConfig=None):

    backend = 'fbgemm' if x86 else 'qnnpack'
    qconfig = torch.quantization.get_default_qconfig(backend)  
    torch.backends.quantized.engine = backend

    print(f"qconfig:{qconfig}")

    if myconfig is not None:
        my_qconfig = torch.quantization.QConfig(
        activation=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_affine),
        weight=MovingAveragePerChannelMinMaxObserver.with_args(qscheme=torch.qint8)
        )
        print(f"my_qconfig:{my_qconfig}")
        qconfig=my_qconfig
    return qconfig

def prepare_data(data_dir,batch_size=32):
    # Prepare the data
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
    ])
    dataset = datasets.CIFAR10(root=data_dir, train=True, download=True,
                                transform=transform)
    # 划分数据集：80% 训练集，20% 测试集
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size,
                                            shuffle=True, num_workers=4)
    return train_loader,test_loader

def train_model(model, dataloader, device,logger,epoches=10):
    # 训练模型
    logger.info(f"Start training in {device}..")
    model.train()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    for epoch in range(epoches):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data[0].to(device), data[1].to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            # print statistics
            running_loss += loss.item()

            if i % 100 == 0:
                logger.info('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' %
                    (epoch+1, epoches, i+1, len(dataloader), loss.item()))
    logger.info('Finished Training and save model to :{}'.format(opt.out_dir))
    torch.save(model, opt.out_dir+'/trian_model.pth')
def evaluate_accuracy(model, dataloader, device,logger):
    correct = 0
    total = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for _,data in tqdm(enumerate(dataloader)):
            images, labels = data[0].to(device), data[1].to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        logger.info('Accuracy of the network on the test images: %d %%' % (
            100 * correct / total))
    return correct / total


def parpare_model(model):
    
    qmodel=QuantModel(model).eval()
    qmodel= torch.load(opt.weights,weights_only=True,map_location='cpu')
    with torch.no_grad():
        qmodel.fuse_model()

    qmodel.qconfig= initialize_calib_method()
    # qat_model=prepare_qat(qmodel)
    return qmodel

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

"""
def sensitive_int8(model):
    # ONE-AT-A-TIME SENSITIVITY ANALYSIS 
    for quantized_layer, _ in model.named_modules():
        print("Only quantizing layer: ", quantized_layer)
        # The module_name key allows module-specific qconfigs. 
        qconfig_dict = {"": None, 
        "module_name":[(quantized_layer, torch.quantization.get_default_qconfig(backend))]}
        
        model_prepared = quantize_fx.prepare_fx(model, qconfig_dict)
        # calibrate
        model_quantized = quantize_fx.convert_fx(model_prepared)
        # evaluate(model)"""


if __name__ == '__main__':
    opt=parse_opt()
    Root=os.getcwd()
    cuda_device='cuda:0'
    cpu_device='cpu'

    print(Root)
    # Set the logger
    logger=set_logger(save_file=os.path.join(Root,r'qat\toturial\qat.log'))

    logger.info('logging Start...')

    # device='cuda:0' if torch.cuda.is_available() else 'cpu'
    # train_dataloader,test_dataloader=prepare_data(opt.data,batch_size=opt.batch_size)
    
    # model=MyModel()
    # logger.info('Model:  ' + str(model))
    # if os.path.exists(opt.out_dir+'/trian_model.pth'):
    #     logger.info('Loading model...')
    #     model_statice=torch.load(opt.weights,map_location='cpu')
    #     model.load_state_dict(model_statice)
    
    # else:
    #     train_model(model,test_dataloader,device=device,logger=logger,epoches=1)
    #     logger.info('Training Done!')
    
    # if opt.evaluation:
    #     logger.info('Evaluating...')
    #     evaluate_accuracy(model,test_dataloader,device='cuda:0',logger=logger)
    # qmodel = parpare_model(model)  # fused model
    # logger.info('Fusing model...')

    # qat_model=prepare_qat(qmodel)
    # print(qmodel)
    

    # model.load_state_dict(torch.load(opt.weights))





