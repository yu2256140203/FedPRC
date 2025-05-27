import torch
import torch.nn as nn
from collections import OrderedDict

def get_model_param_output_channels(model, dummy_input):
    """
    遍历模型中所有叶子模块的前向输出，获取输出通道数，
    并构造一个有序字典：键为参数的全名（模块名+参数名），值为该模块前向输出张量的通道数。
    
    字典顺序按照 model.named_parameters() 的顺序排列。

    参数：
      model: 需要检查的模型，例如 ResNet、VGG 等。
      dummy_input: 用于触发前向传播的虚拟输入张量。

    返回：
      一个 OrderedDict，key 为参数全名（例如 "conv1.weight"），value 为所在模块输出张量的通道数。
    """
    output_channels = {}  # 保存每个模块的输出通道数

    def hook(module, input, output):
        # 处理 output 可能存在多种形式的情况
        if isinstance(output, torch.Tensor):
            shape = output.size()
        elif isinstance(output, (list, tuple)) and len(output) > 0 and isinstance(output[0], torch.Tensor):
            shape = output[0].size()
        else:
            return
        
        # 确保输出张量的 shape 至少为 (N, C, ...)
        if len(shape) >= 2:
            output_channels[module] = shape[1]

    # 为每个叶子模块注册 hook
    hooks = []
    for mod_name, module in model.named_modules():
        if len(list(module.children())) == 0:
            h = module.register_forward_hook(hook)
            hooks.append((mod_name, h))
    
    # 触发一次前向传播以收集数据
    model.eval()
    with torch.no_grad():
        model(dummy_input)
    
    # 移除 hook
    for mod_name, h in hooks:
        h.remove()
    
    # 构造参数全名到输出通道数的映射（无序版本）
    param_channel_dict = {}
    for mod_name, module in model.named_modules():
        if module in output_channels:
            for param_name, _ in module.named_parameters(recurse=False):
                full_param_name = f"{mod_name}.{param_name}" if mod_name else param_name
                param_channel_dict[full_param_name] = output_channels[module]
    
    # 按照 model.named_parameters() 的顺序构造一个有序字典
    ordered_param_channel_dict = OrderedDict()
    for name, _ in model.named_parameters():
        if name in param_channel_dict:
            ordered_param_channel_dict[name] = param_channel_dict[name]
    
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
