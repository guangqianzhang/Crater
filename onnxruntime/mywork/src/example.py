#!
import torch
import torch.nn as nn
import onnx
from onnx import shape_inference
import onnxruntime as ort

import numpy as np

from color_print import print_color,print_error,print_info,print_success,print_warning
class Model(torch.nn.Module):
    def __init__(self, in_channels, out_channels,  bias=False):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=4, kernel_size=3)
        self.bn1   = nn.BatchNorm2d(num_features=4)
        self.act1  = nn.ReLU()
        self.avgpool2d = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool1d = nn.AdaptiveAvgPool1d(1)
        self.linear_inchannels=4
        self.head = nn.Linear(self.linear_inchannels, out_channels, bias)

    def init_weight(self, weights):
        conv_weights= torch.rand(4,4,3,3)
        lieanr_weights = torch.tensor([
            [1, 2, 3, 4],
            [2, 3, 4, 5],
            [3, 4, 5, 6]
        ],dtype=torch.float32)
        with torch.no_grad():
            self.linear.weight.copy_(lieanr_weights)
            self.conv1.weight.copy_(conv_weights)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.avgpool2d(x)
        print(x.shape)
        x = self.head(x.reshape(-1,self.linear_inchannels))
        return x
    



def infer(model,input_data):
    # in_features = torch.tensor([[1, 2, 3, 4],[5,6,7,8]], dtype=torch.float32)
    print_color(f"input shape is: {input_data.shape}", "YELLOW")

    
    x = model(input_data)
    # print("result is: ", x)

    return x

def export_onnx(model,input_data):

    model.eval() #添加eval防止权重继续更新

    # pytorch导出onnx的方式，参数有很多，也可以支持动态size
    # 我们先做一些最基本的导出，从netron学习一下导出的onnx都有那些东西
   
    torch.onnx.export(
        model         = model, 
        args          = (input_data,),
        f             = "../models/convexample.onnx",
        input_names   = ["input0"],
        output_names  = ["output0"],
        opset_version = 12)
    print("Finished onnx export")
    import onnxsim
    model_onnx ,check = onnxsim.simplify(
        "../models/convexample.onnx", 
        check_n=3
    )
    # 下面我们使用onnx自带的shape推理，来修改导出的onnx模型
    onnx_model = onnx.load("../models/convexample.onnx")
    onnx.save(onnx.shape_inference.infer_shapes(onnx_model), "../models/convexample-inferred.onnx")

    

def onnx_infer(input,model_path):

    import onnxruntime as ort
    # input = torch.tensor([[1, 2, 3, 4],[5,6,7,8]], dtype=torch.float32)

    session= ort.InferenceSession(model_path)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    input_shape=session.get_inputs()[0].shape
    if input.shape is not input_shape:
        print_warning(f"input shape is not equal to onnx model input shape {input_shape}, and we will resise it ","RED")
    
        input = input.reshape(input_shape)
    print_color(f"input shape is: {input.shape}", "YELLOW")

    print_color(f"input name is: {input_name}", "YELLOW")
    print_color(f"output name is: {output_name}", "YELLOW")
    
    input_data = {
        input_name: input.numpy()
    }
    output_data = session.run(None, input_data)


    return output_data

def check_out_results(out1, out2):
    print_color("check output results:", "YELLOW")
    if isinstance(out1,torch.Tensor):
        out1= out1.detach().numpy()
    if isinstance(out2,torch.Tensor):
        out2= out2.detach().numpy()

    if (np.abs(out1 - out2) < 0.0001).all():
        print_success("\t \t output result is equal", "GREEN")
    else:
        print_error(f"\t \t output result is not equal. abs error:\
                    {np.mean(np.abs(out1 - out2))}", "RED")
        print_color(f"out1 is: {out1}", "GREEN")
        print_color(f"out2 is: {out2}", "GREEN")




if __name__=="__main__":
    np.random.seed(0)
    torch.manual_seed(0)

    input_data= torch.rand(1, 3, 4, 4, dtype=torch.float32)
    model = Model(3, 3)
    out1=infer(model,input_data)
    export_onnx(model,input_data)
    out2=onnx_infer(input_data,"../models/convexample.onnx")
    check_out_results(out1, out2)
