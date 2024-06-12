# 这是一个关于量化的脚本文件夹

# toturial

+ caculate_scale： torch是如何计算量化因子的
+ export engine： torch是如何导出engine文件，需要安装tensorRT
+ export onnx： torch如何导出onnx文件，需要安装onnx onnxsim 等
+ quantizer：算子量化示例
+ torch_register：量化中的注册回调函数示例
+ quantization：使用torch高级warp量化网络，包括ptq和qat量化内容
+ torch_quantization：使用pytorch_quantization 量化网络，其中使用该工具包灵活替换、量化网络结构，参考bevfusion量化工作。

# others

一些待整理的参考资料

+ export：来自yolov5量化内容
+ quantization aware training ：一个较为完整的qat量化示例，保存为torchscript格式文件。
