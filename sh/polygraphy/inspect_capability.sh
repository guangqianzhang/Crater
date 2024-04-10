#!/bin/bash

onnxmodel=$1
jsonfile="replay.json"
if [[ -z $onnxmodel ]]; then
    echo ">> onnxmodel 为空"
else
    echo ">> onnxmodel 不为空"
# TensorRT 对给定 ONNX 图中 ONNX 操作符支持的详细信息。它还会将原始模型中支持和不支持的子图分区并保存。
    polygraphy inspect capability  $onnxmodel
fi

if [ -e "$jsonfile" ]; then
    echo "tatics repaly jsonfile >>${jsonfile} 文件存在"
    polygraphy inspect tactics $jsonfile
else
    echo "tatic jsonfile >>文件不存在"
fi



