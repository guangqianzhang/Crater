import argparse
import os
import sys
from pathlib import Path
import warnings
import yaml
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.yolo import Model
from utils.dataloaders import create_dataloader
from utils.general import (check_img_size, check_yaml, file_size, colorstr, check_dataset)
from utils.torch_utils import select_device

import py_quant_utils as quant

import val


def collect_stats(model, data_loader, num_batches, device):
    """Feed data to the network and collect statistics"""
    # Enable calibrators
    model.eval()
    for name, module in model.named_modules():
        if isinstance(module, quant.quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    # Feed data to the network for collecting stats
    for i, (image, targets, paths, shapes) in tqdm(enumerate(data_loader), total=num_batches):
        image = image.to(device, non_blocking=True)
        image = image.float()  # uint8 to fp16/32
        image /= 255.0  # 0 - 255 to 0.0 - 1.0
        model(image)
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant.quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()


def compute_amax(model, **kwargs):
    # Load calib result
    for name, module in model.named_modules():
        if isinstance(module, quant.quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                if isinstance(module._calibrator, quant.calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                    module.load_calib_amax(**kwargs)
            print(F"{name:40}: {module}")


def calibrate_model(model, model_name, data_loader, num_calib_batch, calibrator, hist_percentile, out_dir, device):
    """
        Feed data to the network and calibrate.
        Arguments:
            model: detection model
            model_name: name to use when creating state files
            data_loader: calibration data set
            num_calib_batch: amount of calibration passes to perform
            calibrator: type of calibration to use (max/histogram)
            hist_percentile: percentiles to be used for historgram calibration
            out_dir: dir to save state files in
    """

    if num_calib_batch > 0:
        print("Calibrating model")
        with torch.no_grad():
            collect_stats(model, data_loader, num_calib_batch, device)

        if not calibrator == "histogram":
            compute_amax(model, method="max")
            calib_output = os.path.join(out_dir, F"{model_name}-max-{num_calib_batch * data_loader.batch_size}.pth")
            torch.save(model.state_dict(), calib_output)
        else:
            for percentile in hist_percentile:
                print(F"{percentile} percentile calibration")
                compute_amax(model, method="percentile")
                calib_output = os.path.join(out_dir, F"{model_name}-percentile-{percentile}-{num_calib_batch * data_loader.batch_size}.pth")
                torch.save(model.state_dict(), calib_output)

            for method in ["mse", "entropy"]:
                print(F"{method} calibration")
                compute_amax(model, method=method)
                calib_output = os.path.join(out_dir, F"{model_name}-{method}-{num_calib_batch * data_loader.batch_size}.pth")
                torch.save(model.state_dict(), calib_output)


def load_model(weight, device) -> Model:

    model = torch.load(weight, map_location=device)["model"]
    model.float()
    model.eval()

    with torch.no_grad():
        model.fuse()
    return model


def prepare_model(calibrator, opt, device):
    """
    1、Load模型
    2、插入模型的Q和DQ节点
    3、设置量化方法：per_tensor/per_channels,
    4、Dataloader的制作
    """
    with open(opt.data, encoding='utf-8') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  
        data_dict = check_dataset(data_dict)
    calib_path = data_dict['val']

    # 初始化量化方法
    quant.initialize_calib_method(per_channel_quantization=True, calib_method=calibrator)  
    # 加载FP32的Pytorch模型
    model = load_model(opt.weights, device)
    # 为FP32的Torch的Q和DQ的节点
    quant.replace_to_quantization_module(model, ignore_policy=opt.sensitive_layer)
    model.eval()
    model.cuda()


    gs = max(int(model.stride.max()), 32)  # grid size (max stride)
    imgsz, _ = [check_img_size(x, gs) for x in [opt.imgsz, opt.imgsz]]  # verify imgsz are gs-multiples

    # Calib dataloader
    calib_loader = create_dataloader(calib_path,
                                     imgsz,
                                     opt.batch_size,
                                     gs,
                                     hyp=None,
                                     cache=opt.cache,
                                     rect=True,
                                     rank=-1,
                                     workers=opt.workers * 2,
                                     pad=0.5,
                                     prefix=colorstr('calib: '))[0]

    return model, calib_loader


def export_onnx(model, onnx_filename, batch_onnx, dynamic_shape, simplify, imgsz=672, prefix=colorstr('calib: ')):
    from models.yolo import Detect
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Detect):
            m.inplace = False
            m.export = True

    # We have to shift to pytorch's fake quant ops before exporting the model to ONNX
    quant.quant_nn.TensorQuantizer.use_fb_fake_quant = True

    # Export ONNX for multiple batch sizes
    print("Creating ONNX file: " + onnx_filename)
    im = torch.randn(1, 3, imgsz, imgsz)  

    try:
        import onnx
        torch.onnx.export(
            model.cpu() if dynamic_shape else model,  # --dynamic only compatible with cpu
            im.cpu() if dynamic_shape else im,
            onnx_filename,
            verbose=False,
            opset_version=13,
            training= torch.onnx.TrainingMode.EVAL,
            do_constant_folding=True,
            input_names=['images'],
            output_names=['output'],
            dynamic_axes={
                'images': {0: 'batch'},
                'output': {0: 'batch'}  # shape(1,25200,85)
            } if dynamic_shape else None)

        print('ONNX export success, saved as %s' % onnx_filename)

    except ValueError:
        warnings.warn(UserWarning("Per-channel quantization is not yet supported in Pytorch/ONNX RT (requires ONNX opset 13)"))
        print("Failed to export to ONNX")
        return False

    except Exception as e:
            print(f'{prefix} export failure: {e}')
    
    # Checks
    model_onnx = onnx.load(onnx_filename)  # load onnx model
    onnx.checker.check_model(model_onnx)  # check onnx model
    
    # Simplify
    if simplify:
        try:
            import onnxsim
            print(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
            model_onnx, check = onnxsim.simplify(model_onnx)
            assert check, 'assert check failed'
            onnx.save(model_onnx, onnx_filename)
        except Exception as e:
            print(f'{prefix} simplifier failure: {e}')

        print(f'{prefix} export success, saved as {onnx_filename} ({file_size(onnx_filename):.1f} MB)')
        print(f"{prefix} Run ONNX model inference with: 'python detect.py --weights {onnx_filename}'")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=ROOT / 'data/coco128.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', nargs='+', type=str, default=ROOT / 'weights/yolov5n.pt', help='model.pt path(s)')
    parser.add_argument('--model-name', '-m', default='yolov5n', help='model name: default yolov5s')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--imgsz', '--img', '--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--workers', type=int, default=0, help='max dataloader workers (per RANK in DDP mode)')
    parser.add_argument('--cache', type=str, nargs='?', const='ram', help='--cache images in "ram" (default) or "disk"')

    # setting for calibration
    parser.add_argument('--calib-batch-size', type=int, default=64, help='calib batch size: default 64')
    # parser.add_argument('--sensitive-layer', default=['model.24.m.0', 
    #                                                   'model.24.m.1', 
    #                                                   'model.24.m.2'], help='skip sensitive layer: default detect head')
    parser.add_argument('--sensitive-layer', default=[], help='skip sensitive layer: default detect head')
    parser.add_argument('--num-calib-batch', default=64, type=int,
                        help='Number of batches for calibration. 0 will disable calibration. (default: 4)')
    parser.add_argument('--calibrator', type=str, choices=["max", "histogram"], default="max")
    parser.add_argument('--percentile', nargs='+', type=float, default=[99.9, 99.99, 99.999, 99.9999])
    parser.add_argument('--dynamic', default=False, help='dynamic ONNX axes')
    parser.add_argument('--simplify', default=True, help='simplify ONNX file')
    parser.add_argument('--out-dir', '-o', default=ROOT / 'weights/', help='output folder: default ./runs/finetune')
    parser.add_argument('--batch-size-onnx', type=int, default=1, help='batch size for onnx: default 1')

    opt = parser.parse_args()
    opt.data = check_yaml(opt.data)  # check YAML
    return opt


def evaluate_accuracy(model, opt, testloader):
    with open(opt.data, encoding='utf-8') as f:
        data_dict = yaml.load(f, Loader=yaml.SafeLoader)  # data dict

    results, _, _ = val.run(
        data_dict,
        batch_size=opt.batch_size,
        imgsz=opt.imgsz,
        model=model,
        iou_thres=0.65,  # best pycocotools results at 0.65
        single_cls=False,
        dataloader=testloader,
        save_dir=Path(opt.out_dir),
        save_json=False,
        verbose=True,
        plots=True)  # val best model with plots

    map50 = list(results)[3]
    map = list(results)[2]
    return map50, map


def sensitive_analysis(model, opt, data_loader, summary_file='./summary_sensitive_analysis.json'):
    summary = quant.SummaryTool(summary_file)

    # 验证PTQ后的模型精度
    map50_calibrated, map_calibrated = evaluate_accuracy(model, opt, data_loader)
    summary.append([map50_calibrated, map_calibrated, "PTQ"])

    print("Sensitive Analysis by each layer...")
    for i in range(0, len(model.model)):
        layer = model.model[i]
        if quant.have_quantizer(layer):
            print(f"Quantization disable model.{i}")
            quant.disable_quantization(layer).apply()
            map50_calibrated, map_calibrated = evaluate_accuracy(model, opt, data_loader)
            summary.append([map50_calibrated, map_calibrated, f"model.{i}"])
            quant.enable_quantization(layer).apply()
        else:
            print(f"ignore model.{i} because it is {type(layer)}")
    
    summary = sorted(summary.data, key=lambda x:x[0], reverse=True)
    print("Sensitive summary:")
    for n, (map_calibrated, map50_calibrated, name) in enumerate(summary[:10]):
        print(f"Top{n}: Using fp16 {name}, map_calibrated = {map_calibrated:.5f}")


if __name__ == "__main__":
    # 超参数
    opt = parse_opt()
    # 设备选择
    device = select_device(opt.device, batch_size=opt.batch_size)
    # 准备模型和dataloader
    model, data_loader = prepare_model(calibrator=opt.calibrator, opt=opt, device=device)

    # 校准模型
    with torch.no_grad():
        calibrate_model(
            model=model,
            model_name=opt.model_name,
            data_loader=data_loader,
            num_calib_batch=opt.num_calib_batch,
            calibrator=opt.calibrator,
            hist_percentile=opt.percentile,
            out_dir=opt.out_dir,
            device=device)

    sensitive_analysis(model, opt, data_loader)

