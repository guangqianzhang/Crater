#!/bin/bash
# 解析onnx模型结构
onnxmodel=$1
echo "onnxmodel: ${onnxmodel}"
polygraphy inspect model \
           ${onnxmodel}  \
           --show layers 
            # attrs weights