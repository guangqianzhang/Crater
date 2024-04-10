#!/bin/bash
# 生成一个具有动态形状和2个配置文件的引擎
onnxmodel=$1
enginemodel=dynamic_identity.engine
inputdata=inputs.json
outputdata=outputs.json
static_replay=static_replay.json
echo "onnxmodel: ${onnxmodel}"
    polygraphy run ${onnxmodel} --trt \
        --trt-min-shapes X:[1,2,1,1] --trt-opt-shapes X:[1,2,3,3] --trt-max-shapes X:[1,2,5,5] \
        --trt-min-shapes X:[1,2,2,2] --trt-opt-shapes X:[1,2,4,4] --trt-max-shapes X:[1,2,6,6] \
        --save-engine ${enginemodel}\
        --save-inputs $inputdata \
        --save-outputs $outputdata \
        --save-tactics $static_replay
echo ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>"
echo ">>>>>>>>>>>>>>>>inspect model>>>>>>>>"
    polygraphy inspect model ${enginemodel} \
        --show layers

echo ">>>>>>>>>>>>>>>>inspect input>>>>>>>>"
    polygraphy inspect data $inputdata --show-values
echo ">>>>>>>>>>>>>>>>inspect output>>>>>>>>"
    polygraphy inspect data $outputdata --show-values

echo ">>>>>>>>>>>>>>>>inspect tactics>>>>>>>>"
    polygraphy inspect tactics $static_replay --show-values
    # --display-as=trt