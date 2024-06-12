## 说明
文件来自lidar ai iot 中的bevfusion项目:https://github.com/guangqianzhang/Lidar_AI_Solution.git。根据项目算法进行修改，已经可以完成fcos3d算法的ptq量化。
量化内容为backbone和neck，为对head作量化操作

### fcos3dmodule 
    此为尝试重构fcos3d算法在mmdet3d框架中析出部分网络结构，未可验证直接使用

### lean
    此为来自BevFusion加速项目的量化工具包，经修改可以实现fcos3d算法部分结构的量化功能，推理加速约33倍，0.2s。
    主要修改内容为网络结构以及回调钩子函数中的网络推理部分。

    + ptq: fcos3d网络量化脚本
    + runptq_model: 网络推理脚本
    使用时需修改 mmdet3d/apis/inference 中部分代码
    ` 
    def inference_mono_3d_detector(model,cfg, image, ann_file):
    """Inference image with the monocular 3D detector.

    Args:
        model (nn.Module): The loaded detector.
        image (str): Image files.
        ann_file (str): Annotation files.

    Returns:
        tuple: Predicted results and data from pipeline.
    """
    # cfg = model.cfg  # 修改
    device = next(model.parameters()).device  # model device
    `
    + export_onnx: onnx导出脚本，存在“unkown lean”问题。