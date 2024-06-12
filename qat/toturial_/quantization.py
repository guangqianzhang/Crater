import argparse
import copy
import os
import random
import numpy as np
import torch
from tqdm import tqdm

# from utils.train import train_model
from utils.logger import set_logger
from models.mymodel import MyModel,QuantModel
import torch
import torchvision.datasets as datasets
import torchvision.transforms as transforms


from torch.quantization import QuantStub, DeQuantStub, \
    quantize_dynamic, prepare_qat,prepare, convert

def set_random_seeds(random_seed=0):

    torch.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    
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

def train_model(model, dataloader,test_dataloader, device,logger,save_file,epoches=10):
    # 训练模型
    logger.info(f"Start training in {device}..")
    model.train()
    model.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100, 150], gamma=0.1, last_epoch=-1)
    
    for epoch in range(epoches):  # loop over the dataset multiple times
        model.train()
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
            running_loss += loss.item()*inputs.size(0)

            if i % 100 == 0:
                logger.info('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' %
                    (epoch+1, epoches, i+1, len(dataloader), loss.item()))
        train_loss = running_loss / len(dataloader.dataset)

        model.eval()
        eval_loss, eval_accuracy = evaluate_model(model=model, test_loader=test_dataloader, device=device, criterion=criterion)
        scheduler.step()
        logger.info("Epoch: {:03d} Train Loss: {:.3f} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(epoch, train_loss, eval_loss, eval_accuracy))
    logger.info('Finished Training and saver model to :{}'.format(save_file))
    torch.save(model,save_file)
    return model

def evaluate_model(model, test_loader, device, criterion=None):

    model.eval()
    model.to(device)

    running_loss = 0
    running_corrects = 0

    for inputs, labels in test_loader:

        inputs = inputs.to(device)
        labels = labels.to(device)

        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)

        if criterion is not None:
            loss = criterion(outputs, labels).item()
        else:
            loss = 0

        # statistics
        running_loss += loss * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_corrects / len(test_loader.dataset)

    return eval_loss, eval_accuracy

def prepare_model(model,opt:argparse.Namespace):
    def initialize_calib_method(myconfig:torch.quantization.QConfig=None):
        backend = 'fbgemm' if x86 else 'qnnpack'
        if PTQ:
            qconfig = torch.quantization.get_default_qconfig(backend)  
        elif QAT:
            qconfig = torch.quantization.get_default_qat_qconfig(backend)  
        torch.backends.quantized.engine = backend 

        if myconfig is not None:
            qconfig=myconfig
        print(f"qconfig:{qconfig}")
        return qconfig

    
    if os.path.exists(opt.weights):
        logger.info(f'loading weights from {opt.weights}')
        model= torch.load(opt.weights,map_location='cpu')
        # model.load_state_dict(model_static)
    qmodel=QuantModel(model).eval() 
    with torch.no_grad():
        qmodel.fuse_model()

    qmodel.qconfig= initialize_calib_method()
    if PTQ:
        qmodel=prepare(qmodel)
        logger.info(f'PTQ model is prepared: {qmodel}')
    elif QAT:
        qmodel=prepare_qat(qmodel)
        logger.info(f'QAT model is prepared: {qmodel}')
    return qmodel

def calibrate_model(model, loader, device=torch.device("cpu:0")):

    model.to(device)
    model.eval()

    for inputs, labels in tqdm(loader):
        inputs = inputs.to(device)
        labels = labels.to(device)
        _ = model(inputs)

def quantize_model( model,dataloader):
    model.train()
    if PTQ:
        
        model.to(cuda_device)
        calibrate_model(model, dataloader)
        logger.info(f'have quatizaed the model !')
        model=convert(model.eval())
        '''
        RuntimeError: Unsupported qscheme: per_channel_affine
        '''
        logger.info(f'PTQ model is converted: {model}')
    elif QAT:

        train_model(model,test_dataloader,test_dataloader,
                    device=device,
                    logger=logger, 
                    save_file=train_weight_path,
                    epoches=1)

        model=convert(model.eval())

        # Using high-level static quantization wrapper
        # The above steps, including torch.quantization.prepare, calibrate_model, and torch.quantization.convert, are also equivalent to
        # quantized_model = torch.quantization.quantize_qat(model=quantized_model, run_fn=train_model, run_args=[train_loader, test_loader, cuda_device], mapping=None, inplace=False)

        logger.info(f'QAT model is converted: {model}')
    return model


def  parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str,default=r'E:\workdir\data')
    parser.add_argument('--workers', type=int,default=8)
    parser.add_argument('--weights', type=str,default=r'E:\workdir\Crater\qat\toturial\weights\train_model.pth')
    parser.add_argument('--quant-type', type=str,default='int8')
    parser.add_argument('--batch_size', type=int,default=4)
    parser.add_argument('--out-dir','-o', type=str,default=r'E:\workdir\Crater\qat\toturial\weights')
    parser.add_argument('--calib-method', type=str,default='MinMaxObserver')
    parser.add_argument('--evaluation', action='store_true',default=True)

    opt= parser.parse_args()
    return opt


x86=True
QAT= False
PTQ = False if QAT else True


if __name__ == '__main__':
    logger=set_logger(save_dir=os.getcwd()+r'\qat\toturial')
    Root=os.getcwd()
    cuda_device, cpu_device ='cuda:0', 'cpu'
    device='cuda:0' if torch.cuda.is_available() else 'cpu'
    logger.info('device:{}'.format(device))
    opt=parse_opt()
    logger.info(opt)

    quantized_weight_path = os.path.join(Root, 'qat', 'toturial', 'weights', 'quantized_model.pth')
    train_weight_path = os.path.join(Root, 'qat', 'toturial', 'weights', 'train_model.pth')

    # load data
    logger.info("prepared the data loader")
    train_dataloader,test_dataloader=prepare_data(opt.data,batch_size=opt.batch_size)

    model = MyModel()
    logger.debug('defined the model: '+str(model))

    if os.path.exists(quantized_weight_path):
        logger.debug('Exists quantized_model and loading the model...')
    elif os.path.exists(train_weight_path):
        logger.debug('Exists train_model and loading the model...')
        # quant_model=prepare_model(model,opt)
        # logger.info('quantization config: '+str(quant_model.qconfig))

    else:
        # Train the model
        logger.debug('there are not trained mode, start to train the model...')
        train_model(model,test_dataloader,test_dataloader,
                    device=device,
                    logger=logger, 
                    save_file=train_weight_path,
                    epoches=1)
    quant_model= copy.deepcopy(model)
    quant_model=prepare_model(quant_model,opt)
    logger.info('quantization config: '+str(quant_model.qconfig))
    # quantise the model
    quantzed_moidel= quantize_model(quant_model,test_dataloader)
    torch.save(quantzed_moidel, quantized_weight_path)