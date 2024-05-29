import tensorrt as trt
import onnx
import os

onnx_model_path = 'path/to/onnx/model.onnx'
engine_path = 'path/to/tensorrt/engine.plan'

# 创建 TensorRT builder 和 logger
builder = trt.Builder(logger)
network = builder.create_network()

# 创建 ONNX parser
parser = trt.OnnxParser(network, logger)

# 导入 ONNX 模型
with open(onnx_model_path, 'rb') as model:
    if not parser.parse(model.read()):
        for error in range(parser.num_errors):
            print(parser.get_error(error))

# 创建 TensorRT engine
engine = builder.build_cuda_engine(network)

# 保存 TensorRT engine
with open(engine_path, 'wb') as f:
    f.write(engine.serialize())
"""
这个示例代码假设你已经将 ONNX 模型保存到了 path/to/onnx/model.onnx。
它将创建一个 TensorRT builder 和 logger，使用 ONNX parser 将模型导入到 TensorRT 网络中，
然后使用 TensorRT builder 构建 CUDA engine，并将其序列化到 path/to/tensorrt/engine.plan。
在部署推理时，加载 TensorRT engine 并执行推理。可以使用以下代码加载 TensorRT engine 并执行推理：
"""
import tensorrt as trt
import numpy as np

engine_path = 'path/to/tensorrt/engine.plan'

# 加载 TensorRT engine
with open(engine_path, 'rb') as f:
    engine_data = f.read()

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_data)

# 创建 TensorRT context
context = engine.create_execution_context()

# 准备输入数据
input_shape = (batch_size, input_channels, input_height, input_width)
input_data = np.random.randn(*input_shape).astype(np.float32)

# 分配 GPU 内存
input_binding = engine.get_binding_index('input')
input_size = np.product(input_shape) * np.dtype(np.float32).itemsize
input_memory = cuda.mem_alloc(input_size)

# 将输入数据复制到 GPU 内存
cuda.memcpy_htod(input_memory, input_data)

# 分配 GPU 内存用于输出结果
output_binding = engine.get_binding_index('output')
output_shape = tuple(context.get_binding_shape(output_binding))
output_size = np.product(output_shape) * np.dtype(np.float32).itemsize
output_memory = cuda.mem_alloc(output_size)

# 执行推理
bindings = [int(input_memory), int(output_memory)]
context.execute_v2(bindings)

# 将输出结果从 GPU 内存复制回主机内存
output_data = np.zeros(output_shape, dtype=np.float32)
cuda.memcpy_dtoh(output_data, output_memory)
