import argparse
import onnxruntime as ort
import numpy as np
import onnx
import time
from functools import wraps
from memory_profiler import profile
import torch 
import torchvision
import os
def export_onnx(pth_path, torch_model):
    
    torch_model.load_state_dict(torch.load(pth_path,map_location=torch.device('cuda')))
    batch_size=64
    # Input to the model
    first_layer=torch_model.conv1
    x= torch.randn(first_layer.out_channels, first_layer.in_channels, first_layer.kernel_size[0],first_layer.kernel_size[1], requires_grad=False)
    # torch_out = torch_model(x)

    # Export the model
    onnx_path=pth_path.split('.')[-2]+'.onnx'
    if not os.path.exists(onnx_path):
        torch.onnx.export(torch_model,               # model being run
                            x,                         # model input (or a tuple for multiple inputs)
                            onnx_path,   # where to save the model (can be a file or file-like object)
                            export_params=True,        # store the trained parameter weights inside the model file
                            opset_version=10,          # the ONNX version to export the model to
                            do_constant_folding=True,  # whether to execute constant folding for optimization
                            input_names = ['input'],   # the model's input names
                            output_names = ['output'], # the model's output names
                            # dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                            #                 'output' : {0 : 'batch_size'}}
                                        )
        print("Onnx export done!")

def fn_timer(function):
    @wraps(function)
    def function_timer(*args,**kwargs):
        t0=time.time()
        result= function(*args,**kwargs)
        t1=time.time()
        print("Total time running %s: %s seconds" %(function.__name__,str(t1-t0)))
        return result
    return function_timer

@profile
def  onnx_Analyser(onnx_model_path,showWeights=False):
    onnx_model = onnx.load(onnx_model_path)

    # 使用ONNX Runtime创建一个运行提供程序
    onnx_session = ort.InferenceSession(onnx_model_path,providers=['GPUExecutionProvider']) # TensorrtExecutionProvider

    if showWeights:
        # 获取权重参数
        for i, tensor in enumerate(onnx_model.graph.initializer):
            print(f"Weight Parameter {i + 1} Name:", tensor.name)
            print("Weight Parameter Shape:", tensor.dims)
            print("Weight Parameter Data Type:", tensor.data_type)
            # print("Weight Parameter Values:", tensor.raw_data)
            print("\n")

    latency=[]
    # 推理示例
    input={}
    for input_ in onnx_session.get_inputs():
        # 获取模型的输入和输出名称
        input_name = input_.name

        # 打印模型的输入和输出信息
        print("Input Name:", input_name)
        print("Input Shape:", input_.shape)

        input_data = np.random.random( input_.shape).astype(np.float32)
        input.update({input_name:input_data})

    for output_ in onnx_session.get_outputs():
        output_name = output_.name
        print("Output Name:", output_name)
        print("Output Shape:", output_.shape)



    # start=time.time()
    @fn_timer
    def inference(input):
        output = onnx_session.run([output_name], input)
        time.sleep(2)
        return output
    # epoch=10
    # for _ in range(epoch):
    output=inference(input)
    # latency.append(time.time()-start)
    # print("onnx {} Inferencing Time: {} ms".format(onnx_model.graph.name,sum(latency)*1000/len(latency),'.2f'))
    # print("Model Output:", output)

    # onnx_tool.model_profile(onnx_model)
    # MACs:乘加累积操作数
def arg_parse():
    parser = argparse.ArgumentParser(description='ONNX Model Analyser')
    parser.add_argument('--pth', type=str, default="/Lidar_AI_Solution/CUDA-BEVFusion/resnet18-5c106cde.pth", help='path to onnx model')
    parser.add_argument('--onnx', type=str, default="model/resnet50/camera.backbone.onnx", help='path to onnx model')
    args = parser.parse_args()
    print(args)
    return args

if __name__== "__main__":
    args = arg_parse()
    torch_model = torchvision.models.resnet18()
    export_onnx(args.pth,torch_model)
    onnx_Analyser(args.onnx,showWeights=False)
    