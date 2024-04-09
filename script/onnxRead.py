import numpy as np
import onnx
import onnx_tool
import onnxruntime as ort
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "..")))
from utils import fn_timer

import argparse


###########################
# 加载ONNX模型

def  onnx_Analyser(onnx_model_path,showWeights=False):
    onnx_model = onnx.load(onnx_model_path)

    # 使用ONNX Runtime创建一个运行提供程序
    onnx_session = ort.InferenceSession(onnx_model_path,providers=['CPUExecutionProvider'])


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

        input_data = np.random.random((1, *input_.shape[1:])).astype(np.float32)
        input.update({input_name:input_data})

    for output_ in onnx_session.get_outputs():
        output_name = output_.name
        print("Output Name:", output_name)
        print("Output Shape:", output_.shape)


    from memory_profiler import profile
    import time
    start=time.time()
    @fn_timer
    @profile
    def inference(input):
        output = onnx_session.run([output_name], input)
        return output
    output=inference(input)
    latency.append(time.time()-start)
    print("onnx {} Inferencing Time: {} ms".format(onnx_model.graph.name,sum(latency)*1000/len(latency),'.2f'))
    print("Model Output:", output)

    onnx_tool.model_profile(onnx_model)
    # MACs:乘加累积操作数
def arg_parse():
    parser = argparse.ArgumentParser(description='ONNX Model Analyser')
    parser.add_argument('--onnx', type=str, default=r"d:\Files\同步空间\毕业计划BEV\code\modelfile\resnet50int8\head.bbox.onnx", help='path to onnx model')
    args = parser.parse_args()
    return args

if __name__== "__main__":
    args = arg_parse()
    onnx_Analyser(args.onnx,showWeights=False)