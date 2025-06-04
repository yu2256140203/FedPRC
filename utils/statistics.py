import torch
import torch.nn as nn
from collections import OrderedDict
from plato.config import Config
import torch
from collections import OrderedDict
from torch.nn import Module

def get_model_param_output_channels(model, dummy_input=None):
    if Config.data.datasource == "CIFAR10":
        input_sizes = [32, 32]
    elif Config.data.datasource == "CIFAR100":
        input_sizes = [32, 32]
    else:
        input_sizes = [64, 64]
        
    # 实例化模型并准备一个随机输入
    model = model()
    dummy_input = torch.randn(1, 3, input_sizes[0], input_sizes[1])
    
    # 用于存储“模块 -> 输出通道数”
    output_channels = {}
    
    # forward hook：把每个叶子模块（没有子模块的子网络）输出通道数记录下来
    def hook(module: Module, inp, out):
        # 只关心 Tensor 或者包含 Tensor 的 tuple/list
        if isinstance(out, torch.Tensor):
            shape = out.shape
        elif isinstance(out, (list, tuple)) and len(out) > 0 and isinstance(out[0], torch.Tensor):
            shape = out[0].shape
        else:
            return
        
        if len(shape) >= 2:
            output_channels[module] = shape[1]
    
    # 在所有叶子模块上注册 hook
    hooks = []
    for mod_name, module in model.named_modules():
        if len(list(module.children())) == 0:  # 叶子节点
            h = module.register_forward_hook(hook)
            hooks.append(h)
    
    # 触发一次前向传播，收集每个叶子模块的“输出通道数”
    model.eval()
    with torch.no_grad():
        model(dummy_input)
    
    # 卸载所有 hook
    for h in hooks:
        h.remove()
    
    # 构造最终的 OrderedDict：按照 named_modules() 的顺序，
    # 先把每个模块的参数（weight, bias……）塞进去，再把 BN 的 running_mean/var 塞进去
    ordered_param_channel_dict = OrderedDict()
    for mod_name, module in model.named_modules():
        if module not in output_channels:
            continue
        
        # 1) 先加所有 parameters（recurse=False，不递归到子模块）
        for p_name, _ in module.named_parameters(recurse=False):
            full_p_name = f"{mod_name}.{p_name}" if mod_name else p_name
            ordered_param_channel_dict[full_p_name] = output_channels[module]
        
        # 2) 再加所有 buffers（只挑 running_mean 和 running_var）
        for b_name, _ in module.named_buffers(recurse=False):
            if b_name in ("running_mean", "running_var"):
                full_b_name = f"{mod_name}.{b_name}" if mod_name else b_name
                ordered_param_channel_dict[full_b_name] = output_channels[module]
    
    return ordered_param_channel_dict


# 示例用法：
if __name__ == '__main__':
    from torchvision import models
    # 示例中采用 torchvision.models.resnet18
    model = models.resnet18(pretrained=False)
    
    # 对于大多数图像模型，dummy_input 的形状一般为 (batch, 3, 224, 224)
    dummy_input = torch.randn(1, 3, 224, 224)
    
    ordered_param_channels = get_model_param_output_channels(model, dummy_input)
    
    # 按照模型参数顺序打印每个参数名称及其对应的模块输出通道数
    for param_name, channels in ordered_param_channels.items():
        print(f"Parameter '{param_name}': Output channels = {channels}")
