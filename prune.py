import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
from plato.config import Config
from channel_evaluation import st_round
import random



def get_channel_indices(model,hidden_size,submodel_layer_prune_rate=None,input_sizes=[32,32]):
    if submodel_layer_prune_rate == None:
        submodel_layer_prune_rate = 1
    
    input_data = torch.randn(1,3,input_sizes[0],input_sizes[1])
    from utils.statistics import get_model_param_output_channels
    mapping_indices = dict()
    current_out = None
    import numpy as np
    rng = np.random.default_rng(seed=1234)
    modules_indices = get_model_param_output_channels(model,input_data)
    for key,value in modules_indices.items():
        if "conv" in key and current_out == None:
            current_in  =  list(range(3))
            current_out =  list(range(value))
            mapping_indices[key] = (current_in,current_out)
        elif "conv" in key and "short" not in key and "weight" in key:
            last_current_in = current_in
            current_in = current_out
            current_out = sorted(rng.choice(value, int(value * submodel_layer_prune_rate), replace=False).tolist())
            mapping_indices[key] = (current_in,current_out)
        elif "batchnorm" in key or "bn" in key:
            mapping_indices[key] = (current_out)
        elif "linear.weight" in key:
            mapping_indices[key] = (current_in)
        elif "short" in key:
            mapping_indices[key] = (last_current_in,current_out)
        else:
            mapping_indices[key] = (current_out)
    return mapping_indices
        
def get_channel_indices_unuse_hsn(model,submodel_layer_prune_rate=None,importance=None,input_sizes=[32,32]):
    if submodel_layer_prune_rate == None:
        submodel_layer_prune_rate = 1
    input_data = torch.randn(1,3,input_sizes[0],input_sizes[1])
    from utils.statistics import get_model_param_output_channels
    mapping_indices = dict()
    current_out = None
    modules_indices = get_model_param_output_channels(model,input_data)
    for key,value in modules_indices.items():
        if "conv" in key and current_out == None:
            current_in  =  list(range(3))
            current_out =  list(range(value))
            mapping_indices[key] = (current_in,current_out)
        elif "conv" in key and "short" not in key and "weight" in key:
            last_current_in = current_in
            current_in = current_out
            # current_out = sorted(rng.choice(value, int(value * submodel_layer_prune_rate), replace=False).tolist())
            imp = importance[key.rsplit('.', 1)[0]]
            n = int(len(imp) * submodel_layer_prune_rate)
            current_out = sorted(sorted(range(len(imp)), key=lambda i: imp[i], reverse=True)[:n])
            print(current_out)

            mapping_indices[key] = (current_in,current_out)
        elif "batchnorm" in key or "bn" in key:
            mapping_indices[key] = (current_out)
        elif "linear.weight" in key:
            mapping_indices[key] = (current_in)
        elif "short" in key:
            mapping_indices[key] = (last_current_in,current_out)
        else:
            mapping_indices[key] = (current_out)
    return mapping_indices








def hard_threshold_with_ste(x, threshold, beta=100.0):
    soft = torch.sigmoid(beta * (x - threshold))
    hard = (soft >= 0.5).float()
    return hard - soft.detach() + soft
# def prune_mapping_with_global_threshold_and_binary_indices(initial_mapping, final_importance, model, target_ratio=0.5, device='cpu'):
#     # --- 全局统计 ---
#     channels_info = []  # 每项: (layer_name, channel_index, importance, cost)
#     state_dict = model.state_dict()
#     for layer_name, imp in final_importance.items():
#         imp_tensor = final_importance[layer_name].to(device)
#         weight_key = layer_name + ".weight"
#         if weight_key not in state_dict:
#             continue
#         weight = state_dict[weight_key]
#         out_channels = weight.shape[0]
#         for j in range(out_channels):
#             cost = weight[j].numel()
#             bias_key = layer_name + ".bias"
#             if bias_key not in state_dict:
#                 bias_key = layer_name + ".bias"
#             if bias_key in state_dict:
#                 cost += 1
#             channels_info.append((layer_name, j, float(imp_tensor[j].item()), cost))


    
#     total_param = sum([x[3] for x in channels_info])
#     target_keep = total_param * target_ratio
#     channels_info_sorted = sorted(channels_info, key=lambda x: x[2], reverse=True)
#     cumulative = 0
#     cutoff_index = len(channels_info_sorted)
#     for idx, (lname, ch_idx, importance, cost) in enumerate(channels_info_sorted):
#         cumulative += cost
#         if cumulative >= target_keep:
#             cutoff_index = idx
#             break
#     global_threshold = channels_info_sorted[cutoff_index][2] if cutoff_index < len(channels_info_sorted) else 0.0

#     # --- 产生各层的 binary_indices 和 binary_masks ---
#     binary_indices = {}
#     binary_masks = {}
#     for layer_name, imp in final_importance.items():
#         # imp_array = np.array(imp)
#         imp_array = imp

#         keep_idx = np.nonzero(imp_array >= global_threshold)
#         keep_idx = torch.tensor([x[0] for x in keep_idx])
#         if keep_idx.size == 0:
#             keep_idx = np.array([random.randint(0, len(imp_array) - 1)])
#         # 将索引转换为 tensor
#         keep_idx = torch.tensor(keep_idx)


#         # imp_tensor = torch.tensor(imp, device=device) if not torch.is_tensor(imp) else imp.to(device)
#         imp_tensor = final_importance[layer_name].to(device)
#         # 使用 st_round 保持连续性，同时不要 detach 或转换为 numpy
#         mask = hard_threshold_with_ste(imp_tensor, global_threshold, beta=100.0)
#         # 保存 binary_masks 保留 tensor 与梯度信息
#         binary_masks[layer_name] = mask.clone()  # 保留计算图
#         # 同时保存二值索引用于其他目的，可转换为列表
#         binary_indices[layer_name] = np.sort(keep_idx.cpu()).tolist()
#     # print(binary_indices)
    
    




#     # # --- 更新 mapping ---
#     pruned_mapping = initial_mapping.copy()


#     layers = ['layer1', 'layer2', 'layer3', 'layer4']
#     submodel_layer_prune_rates = [
#         [[1,1], [1,1]],   # layer1
#         [[1,1], [1,1]],   # layer2
#         [[1,1], [1,1]],   # layer3
#         [[1,1], [1,1]]    # layer4
#     ]
    
#     hidden_size = Config().parameters.hidden_size
#     # 第一层的输入为原始通道 [0, hidden_size[0]-1]
#     current_input = list(range(hidden_size[0]))
#     pruned_mapping["conv1.weight"] = (list(range(3)),list(range(64)))
#     # print(binary_indices.keys())
#     for layer_idx, layer_name in enumerate(layers):
#         original_out = hidden_size[layer_idx]  # 当前层原始输出通道数
#         block_rates = submodel_layer_prune_rates[layer_idx]
        
#         for block_idx, rates in enumerate(block_rates):
#             r1, r2 = rates
#             prefix = f"{layer_name}.{block_idx}"
            
#             # --- conv1 ---
#             # 基于原始输出通道数计算：用 ceil(original_out * r1)
#             num_conv1 = int(np.ceil(original_out * r1))
#             # 从 0 到 original_out-1 中随机抽取索引
#             pruned_mapping[f"{prefix}.bn1.weight"] = current_input
#             pruned_mapping[f"{prefix}.bn1.bias"]   = current_input
#             conv1_out = binary_indices[f"{prefix}.conv1"]
#             # print(conv1_out)
#             # 对 conv1.weight，输入侧为 current_input（上一层或前面 block 的输出），输出侧为 conv1_out
#             pruned_mapping[f"{prefix}.conv1.weight"] = (current_input, conv1_out)

#             # BN 层（命名为 bn1）紧跟 conv1，这里也采用 conv1_out 作为剪枝结果

            
#             # --- conv2 ---
#             # conv2 的输入为 conv1_out（经过 conv1 剪枝后的输出）
#             # conv2 的输出通道数：基于原始输出通道数，计算 ceil(original_out * r2)
#             num_conv2 = int(np.ceil(original_out * r2))
#             # 安全性判断：不能选取超过 conv1_out 的数量
#             pruned_mapping[f"{prefix}.bn2.weight"] = conv1_out
#             pruned_mapping[f"{prefix}.bn2.bias"]   = conv1_out
#             # 在 conv1_out 这个域中（长度为 num_conv1）抽取 conv2 输出索引
#             conv2_out = binary_indices[f"{prefix}.conv2"]
#             pruned_mapping[f"{prefix}.conv2.weight"] = (conv1_out, conv2_out)
#             # shortcut 映射：输入为 current_input（本 block 输入），输出为 conv2_out
#             # if 'layer1' not in  layer_name and '.1.' not in prefix:
#             pruned_mapping[f"{prefix}.shortcut.weight"] = (current_input, conv2_out)

            
#             # 更新 current_input 为本 block 的输出，用于同层下一个 block
#             current_input = conv2_out
        
#         # 层与层之间，下一层第一个 block 的输入即为这一层最后的 current_input
    
#     # 最后处理 bn4 和 linear 层：采用最后一层原始通道数，并按最后 block 的剪枝率计算

#     final_channels = current_input
#     pruned_mapping['bn4.weight'] = final_channels
#     pruned_mapping['bn4.bias']   = final_channels
#     pruned_mapping['linear.weight'] = final_channels  # 用于全连接层列方向的剪枝
#     pruned_mapping['linear.bias'] = final_channels
#     return pruned_mapping,binary_masks, global_threshold, binary_indices, channels_info_sorted, total_param, target_keep
def compute_effective_total(pruned_mapping, state_dict):
    """
    根据 pruned_mapping（记录每个需要剪枝层的输入/输出索引或通道列表）及 model.state_dict()
    近似计算剪枝后模型的参数总数。这里的规则如下：
      - 如果 mapping[key] 是一个 tuple (in_idx, out_idx) 且对应卷积层（4D权重）：
            effective 参数数 = len(in_idx) * len(out_idx) * (kernel_height * kernel_width)
      - 如果对应的是全连接层（2D权重）：
            effective 参数数 = len(in_idx) * len(out_idx)
      - 对于 BN 层（或其它直接以 list 表示通道索引）的 weight/bias：
            effective 参数数 = len(list)
      - 对于 bias 通常只计算一次。
    """
    import numpy as np
    total = 0
    for key, mapping_val in pruned_mapping.items():
        if key.endswith("weight"):
            # 处理 tuple 类型：表示需要剪枝的层（conv 或 linear）
            if isinstance(mapping_val, tuple):
                in_idx, out_idx = mapping_val
                shape = state_dict.get(key).shape
                if len(shape) == 4:
                    # 卷积层：kernel size 即 shape[2]*shape[3]
                    kernel_size = np.prod(shape[2:])
                    total += len(in_idx) * len(out_idx) * kernel_size
                elif len(shape) == 2:
                    # linear 层
                    total += len(in_idx) * len(out_idx)
                else:
                    # 如果不属于以上两种情况，直接按乘积计算
                    total += len(in_idx) * len(out_idx)
            elif isinstance(mapping_val, list):
                # 例如 BN 层 weight
                total += len(mapping_val)
            # 其他类型暂不处理
        elif key.endswith("bias"):
            if isinstance(mapping_val, tuple):
                # 通常 bias对应输出通道数
                _, out_idx = mapping_val
                total += len(out_idx)
            elif isinstance(mapping_val, list):
                total += len(mapping_val)
    return total


def prune_mapping_with_global_threshold_and_binary_indices(initial_mapping, final_importance, model, target_ratio=0.5, device='cpu'):
    """
    思路说明：
      1. 计算整个模型原始参数数 original_total。
      2. 设 desired_total = original_total * target_ratio。
      3. 利用 final_importance（通常对应卷积层）及一个全局阈值来决定每个可修剪层保留哪些通道，
         并更新 mapping（同时传播剪枝影响到依赖于前一层输出的 BN、linear 等层）。
      4. 写一个辅助函数 compute_pruned_mapping(threshold) 根据阈值计算 binary_indices，并更新 mapping。
      5. 利用二分搜索调整全局阈值，使得根据更新后的 mapping 计算出的 effective 参数数满足 desired_total。
    """
    import numpy as np, torch, random, copy

    # (A) 计算整个模型原始参数总数
    original_total = sum(p.numel() for p in model.parameters())
    desired_total = original_total * target_ratio
    state_dict = model.state_dict()
    
    # 收集所有 final_importance 的数值（假设 final_importance 中的每个值可转化为 numpy 数组）
    all_importance = []
    for imp in final_importance.values():
        # 如果 imp 是 tensor，则取 detach 后的 numpy 数组；因为这里只是用于统计全局最小、最大值
        if isinstance(imp, torch.Tensor):
            imp_np = imp.cpu().detach().numpy()
        else:
            imp_np = np.array(imp)
        all_importance.extend(imp_np.flatten().tolist())

    low_thr = min(all_importance)
    high_thr = max(all_importance)
    
    # 定义辅助函数：给定阈值 threshold，生成 binary_indices、binary_masks，同时计算 probab_masks 时保留梯度
    def compute_pruned_mapping(threshold):
        binary_indices = {}
        binary_masks  = {}
        probab_masks  = {}  # 初始化概率掩码字典
        T = 1.0   # 温度参数（可调整）
        tau = 0.5 # 阈值（可调整）

        # (F) 根据当前阈值生成各层的 mask 与索引
        for layer_name, imp in final_importance.items():
            # 判断 imp 是否为 tensor，如果不是则转换。但这里不做 detach 操作以保留计算图
            if isinstance(imp, torch.Tensor):
                imp_tensor = imp.to(device)
            else:
                imp_tensor = torch.tensor(imp, device=device, dtype=torch.float)
                
            # 为了计算非微分的索引信息，使用 detched 版本
            imp_array = imp_tensor.detach().cpu().numpy()
            keep_idx = np.nonzero(imp_array >= threshold)[0]
            if keep_idx.size == 0:
                keep_idx = np.array([random.randint(0, len(imp_array) - 1)])
            # 记录二值索引，转换为列表（用于后续非微分流程）
            binary_indices[layer_name] = np.sort(keep_idx).tolist()
            # 构造简单的二值 mask（这个部分用于统计，不进行反向传播）
            mask = (imp_array >= threshold).astype(np.float32)
            binary_masks[layer_name] = torch.tensor(mask, device=device)
            # 关键点：利用原始 imp_tensor 计算 probab_masks，以保留梯度与计算图
            probab_masks[layer_name] = torch.sigmoid(imp_tensor * T - tau)
            # loss = sum(probab_masks[layer_name])
            # loss.backward()
            # import time
            # time.sleep(60)
        
        # (G) 根据 binary_indices 更新 mapping
        pruned_mapping = copy.deepcopy(initial_mapping)
        # 以下假设模型架构包含 conv1 和 layer1-layer4，每层含若干 block
        layers = ['layer1', 'layer2', 'layer3', 'layer4']
        submodel_layer_prune_rates = [
            [[1, 1], [1, 1]],   # layer1
            [[1, 1], [1, 1]],   # layer2
            [[1, 1], [1, 1]],   # layer3
            [[1, 1], [1, 1]]    # layer4
        ]
        
        # hidden_size 为每个阶段原始输出通道数（此处假设从 Config 中获得）
        hidden_size = Config().parameters.hidden_size  # 举例: [64, 128, 256, 512]
        # 第一层输入直接使用原始输入通道（比如图像通道数）
        current_input = list(range(hidden_size[0]))
        pruned_mapping["conv1.weight"] = (list(range(3)), list(range(64)))
        
        # 根据 block 传播剪枝索引，依赖关系中“当前层的输出”由 binary_indices 决定
        for layer_idx, layer_name in enumerate(layers):
            original_out = hidden_size[layer_idx]  # 原始输出通道数
            block_rates = submodel_layer_prune_rates[layer_idx]
            for block_idx, rates in enumerate(block_rates):
                r1, r2 = rates
                prefix = f"{layer_name}.{block_idx}"
                
                # --- conv1：输入为 current_input，输出根据对应 final_importance 决定
                conv1_out = binary_indices.get(f"{prefix}.conv1", current_input)
                pruned_mapping[f"{prefix}.bn1.weight"] = current_input
                pruned_mapping[f"{prefix}.bn1.bias"]   = current_input
                pruned_mapping[f"{prefix}.conv1.weight"] = (current_input, conv1_out)
                
                # --- conv2：输入为 conv1_out，输出由 f"{prefix}.conv2" 决定
                conv2_out = binary_indices.get(f"{prefix}.conv2", conv1_out)
                pruned_mapping[f"{prefix}.bn2.weight"] = conv1_out
                pruned_mapping[f"{prefix}.bn2.bias"]   = conv1_out
                pruned_mapping[f"{prefix}.conv2.weight"] = (conv1_out, conv2_out)
                pruned_mapping[f"{prefix}.shortcut.weight"] = (current_input, conv2_out)
                
                # 更新 current_input 为本 block 的输出，供后续 block 使用
                current_input = conv2_out
            # 层间传递：下一层首个 block 的输入为当前层最后的 current_input
        pruned_mapping['bn4.weight'] = current_input
        pruned_mapping['bn4.bias']   = current_input
        pruned_mapping['linear.weight'] = current_input  # 按列剪枝
        pruned_mapping['linear.bias'] = current_input
        
        return pruned_mapping, binary_masks, binary_indices, probab_masks

    # --- (H) 二分搜索求解全局阈值，使更新后的 mapping 对应的模型参数量接近 desired_total ---
    max_iter = 20
    tol = 1e-3  # 可接受的相对误差
    best_mapping = None
    best_thr = None
    best_effective = None
    low = low_thr
    high = high_thr

    for it in range(max_iter):
        mid = (low + high) / 2.0
        mapping_candidate, masks_candidate, indices_candidate, _ = compute_pruned_mapping(mid)
        effective_total = compute_effective_total(mapping_candidate, state_dict)

        # 目标：effective_total 约等于 desired_total
        rel_err = abs(effective_total - desired_total) / desired_total
        # print(f"Iteration {it}: threshold = {mid}, effective_total = {effective_total}, desired = {desired_total}, rel_err = {rel_err}")
        if rel_err < tol:
            best_mapping = mapping_candidate
            best_thr = mid
            best_effective = effective_total
            break
        
        # 如果剪枝后参数数过多，则需要更激进的剪枝（阈值设高一些）
        if effective_total > desired_total:
            low = mid
        else:
            high = mid
        best_mapping = mapping_candidate
        best_thr = mid
        best_effective = effective_total

    # 最后，用 best_thr（全局阈值）确定 binary_masks 和 binary_indices，同时保留 probab_masks 的计算图
    pruned_mapping, binary_masks, binary_indices, probab_masks = compute_pruned_mapping(best_thr)

    return pruned_mapping, binary_masks, probab_masks, binary_indices, 0, 0, 0





def prune_state_dict(full_state, mapping_indices):
    """
    裁剪完整模型状态字典，对每个参数：
    - 若 mapping 为元组，则对卷积层/shortcut同时在第0维（输出通道）和第1维（输入通道）裁剪。
    - 对于 BN 层及 linear，则仅在相应一维裁剪。
    """
    new_state = {}
    for key, param in full_state.items():
        
        if key in mapping_indices:
            mapping = mapping_indices[key]
            # print(key)
            if 'conv' in key or 'shortcut' in key:
                # print("当前进入",key)
                in_indices, out_indices = mapping
                t_in = torch.tensor(in_indices, device=param.device, dtype=torch.long)
                t_out = torch.tensor(out_indices, device=param.device, dtype=torch.long)
                # 对于卷积权重（形状：[out_channels, in_channels, kH, kW]）：

                new_state[key] = param.index_select(0, t_out).index_select(1, t_in).clone()
            elif key == 'linear.weight':
                t_cols = torch.tensor(mapping, device=param.device, dtype=torch.long)
                new_state[key] = param[:, t_cols].clone()
            elif 'bn' in key:
                # print(key)
                # print(param)
                t_idx = torch.tensor(mapping, device=param.device, dtype=torch.long)
                # print(t_idx)
                new_state[key] = param.index_select(0, t_idx).clone()
            else:
                new_state[key] = param.clone()
        else:
            new_state[key] = param.clone()
    # print(new_state["layer1.0.conv1.weight"].shape)
    # print(mapping_indices.keys())
    # print(full_state.keys())
    return new_state

# def client_state_dict(full_state, binary_masks):
#     new_state = {}
#     for key, param in full_state.items():
#         if key in binary_masks:
#             binary_mask = binary_masks[key]  # 这里 binary_mask 为一个 tensor 且保留计算图
#             # 根据层的类别决定如何进行 mask 乘法
#             if 'conv' in key or 'shortcut' in key:
#                 # 卷积层权重形状 [out_channels, in_channels, kH, kW]
#                 # 假定 binary_mask 形状为 [out_channels]
#                 binary_mask_expanded = binary_mask.view(-1, 1, 1, 1)
#                 new_param = param * binary_mask_expanded
#             elif key == 'linear.weight':
#                 # 全连接层权重形状 [out_features, in_features]
#                 # 假定 binary_mask 形状为 [out_features]
#                 binary_mask_expanded = binary_mask.view(-1, 1)
#                 new_param = param * binary_mask_expanded
#             elif 'bn' in key:
#                 # 对于 BatchNorm 层，通常保留参数不进行 mask 调整
#                 new_param = param
#             else:
#                 # 对于其他层，尝试将 mask 扩展到参数张量维度
#                 binary_mask_expanded = binary_mask
#                 while binary_mask_expanded.dim() < param.dim():
#                     binary_mask_expanded = binary_mask_expanded.unsqueeze(-1)
#                 new_param = param * binary_mask_expanded
#         else:
#             # 如果该参数没有对应的 mask，则直接保留
#             new_param = param
#         new_state[key] = new_param
#     return new_state
from collections import OrderedDict

def client_state_dict(model, binary_masks):
    """
    直接利用 model.named_parameters() 构造字典，
    保证返回的参数仍然是 model 内正在参与运算的张量，从而保持计算图。
    """
    new_state = OrderedDict()
    for name, param in model.named_parameters():
        #因为mask没有weight的key，只有层名
        if name.rsplit('.', 1)[0] in binary_masks:
            # print(param.device)
            binary_mask = binary_masks[name.rsplit('.', 1)[0]].to(param.device)  # 应该保证该 mask 是一个保留计算图的 tensor
            # 根据层的类型选择合适的 mask 操作
            if 'conv' in name or 'shortcut' in name:
                binary_mask_expanded = binary_mask.view(-1, 1, 1, 1)
                new_state[name] = param * binary_mask_expanded
            elif name == 'linear.weight':
                binary_mask_expanded = binary_mask.view(-1, 1)
                new_state[name] = param * binary_mask_expanded
            elif 'bn' in name:
                new_state[name] = param  # BatchNorm层保持原参数
            else:
                # 尝试自动扩展 mask 到与 param 同形状
                tmp_mask = binary_mask
                while tmp_mask.dim() < param.dim():
                    tmp_mask = tmp_mask.unsqueeze(-1)
                new_state[name] = param * tmp_mask
        else:
            new_state[name] = param
    return new_state

def restore_pruned_state(full_state, pruned_state, mapping_indices):
    """
    将剪枝后的子模型参数恢复至原始全模型的尺寸。
    对于 mapping_indices 中记录的每个参数：
      - 如果是二维参数（如 conv 的 weight），mapping_indices[key] 为元组 (in_indices, out_indices)；
        此时创建一个与 full_state[key] 尺寸相同的新张量，用 full_state[key] 的值做初始，然后将对应子模型的参数赋值到
        位置 [out_indices, in_indices, ...]；
      - 对于一维参数（例如 BN 层的 weight/bias），mapping_indices[key] 为列表，
        则新张量在这些索引位置赋值子模型参数；
      - 特别地，对于 linear.weight（全连接层的权重），全模型参数形状通常为 (num_classes, original_features)，
        子模型参数形状为 (num_classes, len(selected_features))，因此需要针对第二个维度（列）进行更新。
    对于全模型中没有剪枝的参数，若子模型中存在则直接使用子模型参数，否则使用原始全模型参数。
    """
    restored_state = {}
    # print(mapping_indices.keys())
    for key, full_param in full_state.items():
        if key in mapping_indices:
            # print(key)
            
            mapping = mapping_indices[key]
            restored_param = full_param.clone()
            sub_param = pruned_state[key]
            # if isinstance(mapping, tuple):
            if 'conv' in key or 'shortcut' in key:
                # 针对二维的参数，例如卷积层的 weight，mapping 格式为 (input_indices, output_indices)
                in_idx, out_idx = mapping
                in_idx_tensor = torch.tensor(in_idx, dtype=torch.long, device=full_param.device)
                out_idx_tensor = torch.tensor(out_idx, dtype=torch.long, device=full_param.device)
                # 假设参数 shape 为 (out_channels, in_channels, ...)，利用高级索引覆盖对应位置
                restored_param[out_idx_tensor.unsqueeze(1), in_idx_tensor.unsqueeze(0)] = sub_param
            elif isinstance(mapping, list):
                # 针对一维参数或特殊二维参数（如 linear.weight 需要更新列）
                idx_tensor = torch.tensor(mapping, dtype=torch.long, device=full_param.device)
                # 对于 linear.weight, 需要更新的是列而不是行
                if key.lower().startswith("linear") and "weight" in key:
                    # full_param 的 shape 为 (num_classes, original_features)，子模型参数为 (num_classes, len(mapping))
                    restored_param[:, idx_tensor] = sub_param
                elif key.lower().startswith("linear") and "bias" in key:
                    # 对于 linear.bias，不进行索引操作，直接使用子模型参数
                    restored_param = sub_param.clone()
                else:
                    restored_param[idx_tensor] = sub_param
            restored_state[key] = restored_param
        else:
            restored_state[key] = pruned_state[key] if key in pruned_state else full_param.clone()
    return restored_state


def aggregate_submodel_states(full_state, sub_state_list, mapping_indices_list):
    """
    聚合多个子模型参数。
    对于每个子模型，先调用 restore_pruned_state 将其参数恢复到与 full_state 相同的尺寸（缺失部分默认使用原始全模型参数）。
    然后对所有子模型恢复后的状态字典进行逐参数、逐元素平均（例如取均值）。
    返回聚合后的全模型状态字典 aggregated_state。
    """
    restored_states = []
    for pruned_state, mapping_indices in zip(sub_state_list, mapping_indices_list):
        restored_state = restore_pruned_state(full_state, pruned_state, mapping_indices)
        restored_states.append(restored_state)
    
    aggregated_state = {}
    for key in full_state.keys():
        # 对所有恢复后的状态对应 key 的参数逐元素求平均，确保形状一致
        agg_param = sum(state[key] for state in restored_states) / len(restored_states)
        aggregated_state[key] = agg_param
    return aggregated_state,restored_states







def reverse_prune_rates(mapping_indices, hidden_size):
    """
    逆推出 submodel_prune_rates（仅针对子模型的卷积层部分），
    根据 mapping_indices 中每层每个 block 的保留通道数量恢复出有效剪枝率。
    
    对每个层、每个 block：
      - 对于 conv1，mapping_indices 的键 "{layer}.{block}.conv1.weight" 存储了 (prev_input, conv1_out)；
        有效 r1 = len(conv1_out) / hidden_size[layer_idx]
      - 对于 conv2，对应的键 "{layer}.{block}.conv2.weight" 存储了 (conv1_out, conv2_out)；
        有效 r2 = len(conv2_out) / hidden_size[layer_idx]
    
    返回的数据结构为一个列表（按层顺序），每个元素为该层中各 block 的 [r1, r2] 列表。
    """
    submodel_prune_rates = []
    layers = ['layer1', 'layer2', 'layer3', 'layer4']
    for layer_idx, layer_name in enumerate(layers):
        layer_prune_rates = []
        block_idx = 0
        while True:
            key_conv1 = f"{layer_name}.{block_idx}.conv1.weight"
            key_conv2 = f"{layer_name}.{block_idx}.conv2.weight"
            if key_conv1 not in mapping_indices or key_conv2 not in mapping_indices:
                break
            # 对于 conv1，其映射格式为 (prev_input, conv1_out)
            _, conv1_out = mapping_indices[key_conv1]
            # 对于 conv2，其映射格式为 (conv1_out, conv2_out)
            _, conv2_out = mapping_indices[key_conv2]
            # 这里以原始层通道数（hidden_size[layer_idx]）作为基数
            original_channels = hidden_size[layer_idx]
            r1_eff = len(conv1_out) / original_channels
            r2_eff = len(conv2_out) / original_channels
            layer_prune_rates.append([r1_eff, r2_eff])
            block_idx += 1
        submodel_prune_rates.append(layer_prune_rates)
    
    return submodel_prune_rates


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

###############################################################
# 3. 主流程：完整模型预训练、子模型构造、训练、聚合与测试
###############################################################

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载 CIFAR10 数据集（训练集和测试集均适用，不变）
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset  = CIFAR10(root="./data", train=False, download=True, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader   = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    criterion = nn.CrossEntropyLoss()
    
    # 定义完整模型和子模型的剪枝率配置
    full_layer_prune_rates = [
        [[1, 1], [1, 1]],  # layer1
        [[1, 1], [1, 1]],  # layer2
        [[1, 1], [1, 1]],  # layer3
        [[1, 1], [1, 1]]   # layer4
    ]
    submodel1_prune_rates = [
        [[0.5, 0.25], [0.5, 0.5]],  # layer1
        [[0.5, 0.5], [0.5, 0.5]],  # layer2
        [[0.5, 0.5], [0.5, 0.5]],  # layer3
        [[0.5, 0.5], [0.5, 0.5]]   # layer4
    ]
    submodel2_prune_rates = [
        [[0.33, 0.33], [0.33, 0.33]],  # layer1
        [[0.33, 0.33], [0.33, 0.33]],  # layer2
        [[0.33, 0.33], [0.33, 0.33]],  # layer3
        [[0.33, 0.33], [0.33, 0.33]]   # layer4
    ]
    hidden_size = [64, 128, 256, 512]
    
    num_epochs_full = 1  # 完整模型预训练 epoch 数（仅为演示）
    num_epochs_sub  = 1  # 子模型训练 epoch 数
    from resnet import resnet18  # 假设 resnet18 定义中使用剪枝率配置构造网络
    model = resnet18(full_layer_prune_rates, track=False).to(device) 
    print("全模型参数量:", count_parameters(model))
    print(model)
    
    for itr in range(5):
        print("======================================")
        print(f"【第 {itr+1} 次迭代】")
        
        # -------------------------------
        # (1) 预训练完整模型
        # -------------------------------

        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        model.train()
        for epoch in range(num_epochs_full):
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad()
                outputs = model(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"预训练 Epoch {epoch}: Loss {total_loss/len(train_loader):.4f}")
        
        # 测试预训练的完整模型
        model.eval()
        correct_full, total_full = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                total_full += y.size(0)
                correct_full += (predicted == y).sum().item()
        full_acc = 100 * correct_full / total_full
        print(f"预训练全模型测试准确率: {full_acc:.2f}%")
        
        # 保存预训练全模型参数
        full_state = model.state_dict()
        # print(full_state.keys())
        
        # -------------------------------
        # (2) 构造并训练子模型
        # -------------------------------
        # 根据子模型剪枝率计算剪枝索引，并裁剪出子模型参数
        mapping_indices_sub1 = get_channel_indices(hidden_size, submodel1_prune_rates)
        mapping_indices_sub2 = get_channel_indices(hidden_size, submodel2_prune_rates)
        submodel1_state = prune_state_dict(full_state, mapping_indices_sub1)
        submodel2_state = prune_state_dict(full_state, mapping_indices_sub2)

        # 逆推出有效的剪枝率
        reversed_prune_rates = reverse_prune_rates(mapping_indices_sub1, hidden_size)
        # print("\n逆推出的 submodel1_prune_rates (基于保留通道比例):")
        # for layer in reversed_prune_rates:
            # print(layer)
        
        # 构造子模型，并加载裁剪后的参数
        submodel1 = resnet18(submodel1_prune_rates, track=False).to(device)
        # print(submodel1)
        submodel2 = resnet18(submodel2_prune_rates, track=False).to(device)
        # print(submodel2)
        submodel1.load_state_dict(submodel1_state, strict=False)
        submodel2.load_state_dict(submodel2_state, strict=False)
        print("子模型1参数量:", count_parameters(submodel1))
        print("子模型2参数量:", count_parameters(submodel2))
        
        # 数据集拆分：将训练集一分为二（索引前半部分给子模型1，后半部分给子模型2）
        train_size = len(train_dataset)
        indices = list(range(train_size))
        split = train_size // 2
        indices_sub1 = indices[:split]
        indices_sub2 = indices[split:]
        subtrain_loader1 = DataLoader(train_dataset, batch_size=32,
                                      sampler=torch.utils.data.SubsetRandomSampler(indices_sub1))
        subtrain_loader2 = DataLoader(train_dataset, batch_size=32,
                                      sampler=torch.utils.data.SubsetRandomSampler(indices_sub2))
        
        optimizer1 = optim.Adam(submodel1.parameters(), lr=0.001)
        optimizer2 = optim.Adam(submodel2.parameters(), lr=0.001)
        
        print("开始训练子模型1...")
        submodel1.train()
        for epoch in range(num_epochs_sub):
            total_loss = 0
            for x, y in subtrain_loader1:
                x, y = x.to(device), y.to(device)
                optimizer1.zero_grad()
                outputs = submodel1(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer1.step()
                total_loss += loss.item()
            print(f"子模型1 Epoch {epoch}: Loss {total_loss/len(subtrain_loader1):.4f}")
        
        print("开始训练子模型2...")
        submodel2.train()
        for epoch in range(num_epochs_sub):
            total_loss = 0
            for x, y in subtrain_loader2:
                x, y = x.to(device), y.to(device)
                optimizer2.zero_grad()
                outputs = submodel2(x)
                loss = criterion(outputs, y)
                loss.backward()
                optimizer2.step()
                total_loss += loss.item()
            print(f"子模型2 Epoch {epoch}: Loss {total_loss/len(subtrain_loader2):.4f}")
        
        # 测试子模型
        submodel1.eval()
        correct_sub1, total_sub1 = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = submodel1(x)
                _, predicted = torch.max(outputs, 1)
                total_sub1 += y.size(0)
                correct_sub1 += (predicted == y).sum().item()
        sub1_acc = 100 * correct_sub1 / total_sub1
        print(f"子模型1测试准确率: {sub1_acc:.2f}%")
        
        submodel2.eval()
        correct_sub2, total_sub2 = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = submodel2(x)
                _, predicted = torch.max(outputs, 1)
                total_sub2 += y.size(0)
                correct_sub2 += (predicted == y).sum().item()
        sub2_acc = 100 * correct_sub2 / total_sub2
        print(f"子模型2测试准确率: {sub2_acc:.2f}%")
        
        # -------------------------------
        # (3) 聚合子模型参数并更新全模型
        # -------------------------------
        submodel1_state_trained = submodel1.state_dict()
        submodel2_state_trained = submodel2.state_dict()
        mapping_indices_list = [mapping_indices_sub1, mapping_indices_sub2]
        sub_state_list = [submodel1_state_trained, submodel2_state_trained]
        
        aggregated_state = aggregate_submodel_states(full_state, sub_state_list, mapping_indices_list)
        model.load_state_dict(aggregated_state, strict=False)
        
        # 测试聚合后全模型
        model.eval()
        correct_agg, total_agg = 0, 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                _, predicted = torch.max(outputs, 1)
                total_agg += y.size(0)
                correct_agg += (predicted == y).sum().item()
        agg_acc = 100 * correct_agg / total_agg
        print(f"聚合后全模型测试准确率: {agg_acc:.2f}%")
        
        # -------------------------------
        # (4) 验证聚合是否成功
        # 对一个测试 batch，比对聚合模型输出与两个子模型输出均值的平均差异
        # -------------------------------
        dataiter = iter(test_loader)
        images, labels = next(dataiter)
        images, labels = images.to(device), labels.to(device)
        with torch.no_grad():
            output_agg = model(images)
            output_sub1 = submodel1(images)
            output_sub2 = submodel2(images)
            output_avg = (output_sub1 + output_sub2) / 2.0
            diff = torch.abs(output_agg - output_avg).mean().item()
        print(f"聚合模型与子模型均值输出的平均差异: {diff:.4f}")
        
        print("======================================\n")
