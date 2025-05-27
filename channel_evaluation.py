import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, ToTensor, Normalize
from torch.utils.data import DataLoader
import numpy as np
import resnet  # 请确保你的 resnet 模块中包含 resnet18 的定义
import matplotlib.pyplot as plt
import os
import math

# 如果 img 文件夹不存在则创建
if not os.path.exists("img"):
    os.makedirs("img")

##############################################
# 0. STE (Straight Through Estimator) 实现
##############################################
class StRound(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        # 使用 STE，直接将梯度传递
        return grad_output

def st_round(x):
    return StRound.apply(x)

##############################################
# 1. 通道重要性评估模块 (使用 forward hook)
##############################################
class ChannelImportanceEvaluator:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.score_dict = {}
        self.threshold = None
        self.current_orig = None
        self.current_binary = None

    def hook_fn(self, name):
        def _hook(module, inputs, output):
            target_size = (32, 32)
            up_out = F.interpolate(output, size=target_size, mode="bilinear", align_corners=False)
            binary_feat = (up_out > self.threshold).float()
            binary_orig = self.current_binary.expand_as(binary_feat)
            B, C, H, W = binary_feat.shape
            inter = (binary_feat * binary_orig).view(B, C, -1).sum(dim=2)
            union = (((binary_feat + binary_orig) > 0).float()).view(B, C, -1).sum(dim=2)
            iou = inter / (union + 1e-6)
            A_score = iou.mean(dim=0)
            norm_sq = output.pow(2).sum(dim=(2, 3))
            var_orig = self.current_orig.var(dim=(2, 3), unbiased=False).mean(dim=1, keepdim=True)
            B_score = (norm_sq / (var_orig + 1e-6)).mean(dim=0)
            if name not in self.score_dict:
                self.score_dict[name] = {"A_sum": A_score.clone(), "B_sum": B_score.clone(), "count": 1}
            else:
                self.score_dict[name]["A_sum"] += A_score
                self.score_dict[name]["B_sum"] += B_score
                self.score_dict[name]["count"] += 1
        return _hook

    # def evaluate(self, data_loader, threshold=0.5):
    #     self.threshold = threshold
    #     hooks = []
    #     for name, module in self.model.named_modules():
    #         if isinstance(module, nn.Conv2d):
    #             h = module.register_forward_hook(self.hook_fn(name))
    #             hooks.append(h)
    #     self.model.eval()
    #     with torch.no_grad():
    #         for x, _ in data_loader:
    #             x = x.to(self.device)
    #             self.current_orig = x.clone()
    #             gray = x.mean(dim=1, keepdim=True)
    #             self.current_binary = F.interpolate(gray, size=(32, 32), mode="bilinear", align_corners=False)
    #             self.current_binary = (self.current_binary > threshold).float()
    #             _ = self.model(x)
    #     for h in hooks:
    #         h.remove()
    #     importance_dict = {}
    #     for name, scores in self.score_dict.items():
    #         count = scores["count"]
    #         A_avg = scores["A_sum"] / count
    #         B_avg = scores["B_sum"] / count
    #         gamma = 0.2
    #         A_transformed = torch.pow(A_avg, gamma)
    #         B_transformed = torch.pow(B_avg, gamma)
    #         norm_A = (A_transformed - A_transformed.min()) / (A_transformed.max() - A_transformed.min() + 1e-6)
    #         norm_B = (B_transformed - B_transformed.min()) / (B_transformed.max() - B_transformed.min() + 1e-6)
    #         importance = norm_A * norm_B
    #         importance_dict[name] = importance.detach().cpu().numpy()
    #     return importance_dict
    def evaluate(self, data_loader, threshold=0.5):
        self.threshold = threshold
        hooks = []
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                h = module.register_forward_hook(self.hook_fn(name))
                hooks.append(h)
        self.model.eval()
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(self.device)
                self.current_orig = x.clone()
                gray = x.mean(dim=1, keepdim=True)
                self.current_binary = F.interpolate(gray, size=(32, 32), mode="bilinear", align_corners=False)
                self.current_binary = (self.current_binary > threshold).float()
                _ = self.model(x)
        for h in hooks:
            h.remove()
            
        importance_dict = {}
        for name, scores in self.score_dict.items():
            count = scores["count"]
            A_avg = scores["A_sum"] / count
            B_avg = scores["B_sum"] / count
            gamma = 0.2
            
            # 对 A_avg 与 B_avg 经过 gamma 变换
            A_transformed = torch.pow(A_avg, gamma)
            B_transformed = torch.pow(B_avg, gamma)
            
            # --- 对 A_transformed 使用 Z 分数标准化 ---
            mean_A = A_transformed.mean()
            std_A = A_transformed.std() + 1e-6  # 避免除 0
            norm_A = (A_transformed - mean_A) / std_A
            
            # --- 对 B_transformed 使用 Z 分数标准化 ---
            mean_B = B_transformed.mean()
            std_B = B_transformed.std() + 1e-6
            norm_B = (B_transformed - mean_B) / std_B
            
            # 结合两个归一化项得到重要性指标
            importance = norm_A * norm_B
            
            # 可选：对 importance 再次做 Z 分数标准化，
            # 使得每层的重要性值具有零均值和单位标准差。
            mean_imp = importance.mean()
            std_imp = importance.std() + 1e-6
            z_score_importance = (importance - mean_imp) / std_imp
            
            importance_dict[name] = z_score_importance.detach().cpu().numpy()
            
        return importance_dict


##############################################
# 2. 超结构网络 (HyperStructure Network, HSN) —— 加入 GRU 与 BatchNorm
##############################################
class HyperStructureNetwork(nn.Module):
    def __init__(self, num_layers, d, gru_hidden_dim, h_dim, p=0.5):
        """
        参数：
          num_layers: 需要控制的层数，每层生成一个标量控制值
          d: 每个层的嵌入向量维度
          gru_hidden_dim: GRU 的隐藏状态维度
          h_dim: MLP 隐层的维度
          p: 期望的平均控制值（用于正则化）
          tau: 温度参数，用于缩放 MLP 输出
        """
        super().__init__()
        self.num_layers = num_layers
        self.p = p

        # 定义可学习嵌入，建议初始化幅度较小
        self.embeddings = nn.Parameter(torch.randn(num_layers, d) * 0.1)
        # self.embeddings.requires_grad = False
        # GRU 用于捕获各层间顺序信息。注意 batch_first=False，因此输入形状为 (seq_len, batch, d)
        self.gru = nn.GRU(input_size=d, hidden_size=gru_hidden_dim, batch_first=False)

        # 通过 MLP 将 GRU 输出映射到标量，这里加入 BatchNorm 稳定激活分布
        self.mlp = nn.Sequential(
            nn.Linear(gru_hidden_dim, h_dim),
            nn.BatchNorm1d(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, 1)
        )

    def forward(self):
        # 将 embeddings 转换为序列，形状: (num_layers, 1, d)
        emb_seq = self.embeddings.unsqueeze(1)
        # 经过 GRU，输出 shape: (num_layers, 1, gru_hidden_dim)
        gru_out, _ = self.gru(emb_seq)
        gru_out = gru_out.squeeze(1)  # shape: (num_layers, gru_hidden_dim)
        logits = self.mlp(gru_out)    # shape: (num_layers, 1)
        # 这里可以尝试其他激活方式，例如先使用 tanh 再线性映射到 [0,1]
        v = torch.sigmoid(logits)
        reg_loss = torch.mean((v - self.p) ** 2)
        # c = math.log(self.p / (1 - self.p))  # 在__init__中计算并存储
        # reg_loss = torch.mean((logits - c) ** 2)
        # 将每层的输出放入列表中
        v_list = [v[i] for i in range(self.num_layers)]
        return v_list, reg_loss

##############################################
# 3. 将重要性与超网络输出结合生成全局重要性
##############################################
# def combine_importance(importance_dict, hsn_outputs, device, conv_names):
#     final_importance = {}
#     for i, name in enumerate(conv_names):
#         hsn_val = hsn_outputs[i].to(device)  # shape: (1,)
#         model_imp = torch.tensor(importance_dict[name], device=device)  # shape: (num_channels,)
#         final_imp = hsn_val * model_imp  # 利用广播机制
#         final_importance[name] = final_imp.detach().cpu().numpy()
#     return final_importance
def combine_importance(importance_dict, hsn_outputs, device, conv_names):
    final_importance = {}
    for i, name in enumerate(conv_names):
        # 从超网络获得的输出，要求该张量已经在计算图中，具有 requires_grad=True
        hsn_val = hsn_outputs[i].to(device)  # shape: (1,)
                # 检查 hsn_outputs[i] 是否需要梯度
        # 如果 importance_dict[name] 来自于外部常量，可以选择设置 requires_grad，如果需要的话
        model_imp = torch.tensor(importance_dict[name], device=device, dtype=torch.float)
        # 这里通过广播机制计算最终重要性
        final_imp = hsn_val * model_imp  
        # 将结果直接存入字典（不调用 detach、.cpu() 或 .numpy()）
        final_importance[name] = final_imp
        # def print_grad(grad):
        #     print("hsn_outputs gradient:", grad)

        # # 为 hsn_outputs[i] 注册 hook
        # handle = hsn_outputs[i].register_hook(print_grad)


    return final_importance

##############################################
# 4. 根据全局参数预算进行二值化（生成 mask）
##############################################
# def global_binarize_by_param_budget(final_importance, model, target_param_keep_ratio=0.5, device='cpu'):
#     channels_info = []
#     state_dict = model.state_dict()
#     for layer_name, imp in final_importance.items():
#         imp_tensor = torch.tensor(imp, device=device) if not torch.is_tensor(imp) else imp.to(device)
#         weight_key = layer_name + ".weight"
#         if weight_key not in state_dict:
#             continue
#         weight = state_dict[weight_key]
#         out_channels = weight.shape[0]
#         for j in range(out_channels):
#             cost = weight[j].numel()
#             bias_key = layer_name + ".bias"
#             if bias_key in state_dict:
#                 cost += 1
#             channels_info.append((layer_name, j, imp_tensor[j].item(), cost))
#     total_param = sum(x[3] for x in channels_info)
#     target_keep = total_param * target_param_keep_ratio
#     channels_info_sorted = sorted(channels_info, key=lambda x: x[2], reverse=True)
#     cumulative = 0
#     cutoff_index = len(channels_info_sorted)
#     for idx, (lname, ch_idx, importance, cost) in enumerate(channels_info_sorted):
#         cumulative += cost
#         if cumulative >= target_keep:
#             cutoff_index = idx
#             break
#     threshold = channels_info_sorted[cutoff_index][2] if cutoff_index < len(channels_info_sorted) else 0.0

#     binary_indices = {}
#     sorted_positions = {}
#     for layer_name, imp in final_importance.items():
#         imp_tensor = torch.tensor(imp, device=device) if not torch.is_tensor(imp) else imp.to(device)
#         # 获取大于等于阈值的通道索引
#         mask_indices = torch.argsort(imp_tensor, descending=True)
#         mask_indices = mask_indices[imp_tensor[mask_indices] >= threshold]
#         binary_indices[layer_name] = mask_indices.detach().cpu().numpy()
#         sorted_positions[layer_name] = mask_indices.detach().cpu().numpy()
#         #已经修改成每层的索引了
#     return binary_indices, threshold, sorted_positions, channels_info_sorted, total_param, target_keep
def global_binarize_by_param_budget(final_importance, model, target_param_keep_ratio=0.5, device='cpu'):
    channels_info = []
    state_dict = model.state_dict()
    for layer_name, imp in final_importance.items():
        imp_tensor = torch.tensor(imp, device=device) if not torch.is_tensor(imp) else imp.to(device)
        weight_key = layer_name + ".weight"
        if weight_key not in state_dict:
            continue
        weight = state_dict[weight_key]
        out_channels = weight.shape[0]
        for j in range(out_channels):
            cost = weight[j].numel()
            bias_key = layer_name + ".bias"
            if bias_key in state_dict:
                cost += 1
            channels_info.append((layer_name, j, imp_tensor[j].item(), cost))
    total_param = sum(x[3] for x in channels_info)
    target_keep = total_param * target_param_keep_ratio
    channels_info_sorted = sorted(channels_info, key=lambda x: x[2], reverse=True)
    cumulative = 0
    cutoff_index = len(channels_info_sorted)
    for idx, (lname, ch_idx, importance, cost) in enumerate(channels_info_sorted):
        cumulative += cost
        if cumulative >= target_keep:
            cutoff_index = idx
            break
    threshold = channels_info_sorted[cutoff_index][2] if cutoff_index < len(channels_info_sorted) else 0.0

    binary_indices = {}
    binary_masks = {}
    sorted_positions = {}
    for layer_name, imp in final_importance.items():
        imp_tensor = torch.tensor(imp, device=device) if not torch.is_tensor(imp) else imp.to(device)
        # 直接使用布尔筛选，保留原来顺序的索引
        mask_indices_unsorted = torch.nonzero(imp_tensor >= threshold).view(-1)
        binary_indices[layer_name] = mask_indices_unsorted.detach().cpu().numpy()
        mask = st_round((imp_tensor >= threshold).float())
        binary_masks[layer_name] = mask.detach().cpu().numpy()
        # 如果需要，也可以保留重要性从高到低排序的索引作为排序版本
        sorted_positions[layer_name] = torch.argsort(imp_tensor, descending=True).detach().cpu().numpy()
    for layer_name in binary_masks:
        binary_masks[layer_name].requires_grad = True


    return binary_indices,binary_masks, threshold, sorted_positions, channels_info_sorted, total_param, target_keep




##############################################
# 6. 带动态 Mask 的模型（修改版）
##############################################
# class MaskedResNet(nn.Module):
#     def __init__(self, base_model, conv_names, hyper_net, channel_importance,bin_threshold,random):
#         super().__init__()
#         self.base_model = base_model
#         self.conv_names = conv_names
#         self.hyper_net = hyper_net
#         # 保存从评估器获得的各层通道重要性
#         self.channel_importance = channel_importance
#         self.bin_threshold = bin_threshold
#         self.beta = 100.0  # 温度参数，用于 sigmoid 近似硬阈值
#         self.random = random

#     def forward(self, x, update=False):
#         # 获得超网络输出（每层标量控制值）和正则化损失
#         v_list, reg_loss = self.hyper_net()
#         # 将超网络输出与对应卷积层名称一一对应（假设 conv_names 有序）
#         self.hsn_outputs = {name: v for name, v in zip(self.conv_names, v_list)}
#         if self.random:
#             self.hsn_outputs = {name: 1 for name in self.conv_names}



#         def hook_fn(module, inputs, output, name):
#             # 获取该层预评估得到的通道重要性
#             layer_imp_value = torch.tensor(self.channel_importance[name], device=output.device)
#             # 如果当前层是第一层卷积，则 mask 固定为全 1，不进行剪枝
#             # if name == self.conv_names[0]:
#             #     binary_mask = torch.ones_like(layer_imp_value)

#             final_imp = self.hsn_outputs[name] * layer_imp_value
#             binary_mask = st_round((final_imp >= self.bin_threshold).float())
#             mask_expanded = binary_mask.view(1, -1, 1, 1)
            
#             return output * mask_expanded

#         self.hooks = []
#         # 对 base_model 中属于 conv_names 的模块注册 hook
#         for name, module in self.base_model.named_modules():
#             if name in self.conv_names:
#                 hook = module.register_forward_hook(lambda mod, inp, out, n=name: hook_fn(mod, inp, out, n))
#                 self.hooks.append(hook)
#         out = self.base_model(x)
#         for h in self.hooks:
#             h.remove()
#         if update:
#             return out, reg_loss
#         return out

class MaskedResNet(nn.Module):
    def __init__(self, base_model, probab_masks):
        super().__init__()
        self.base_model = base_model
        self.probab_masks = probab_masks
        self.conv_names = self.probab_masks.keys()

    def forward(self, x, update=False):
        def hook_fn(module, inputs, output, name):
            mask_expanded = self.probab_masks[name].reshape(1, -1, 1, 1).clone()
            

            output = output * mask_expanded

            return output
        
        self.hooks = []
        # 对 base_model 中属于 conv_names 的模块注册 hook
        for name, module in self.base_model.named_modules():
            if name in self.conv_names:
                hook = module.register_forward_hook(lambda mod, inp, out, n=name: hook_fn(mod, inp, out, n))
                self.hooks.append(hook)
        out = self.base_model(x)
        for h in self.hooks:
            h.remove()
        if update:
            return out, reg_loss
        return out

##############################################
# 7. Main：整体流程 —— 数据加载、评估、超网络训练、剪枝、测试及可视化
##############################################
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # 构造 ResNet18 模型，剪枝率均设为 1，仅用于测试评估及 Mask 流程
    layer_prune_rates = [
        [[1, 1], [1, 1]],   # layer1
        [[1, 1], [1, 1]],   # layer2
        [[1, 1], [1, 1]],   # layer3
        [[1, 1], [1, 1]]    # layer4
    ]
    base_model = resnet.resnet18(layer_prune_rates, track=False)
    base_model = base_model.to(device)

    # 加载 CIFAR10 数据集
    transform = Compose([ToTensor(), Normalize((0.5,), (0.5,))])
    test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(base_model.parameters()), lr=0.001)
    num_epochs = 0

    print("Starting pre-training...")
    base_model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = base_model(x)
            loss_cls = criterion(outputs, y)
            lam = 500
            loss = loss_cls
            loss.backward()
            optimizer.step()
            total_loss += loss.item()


    # 1. 评估各卷积层重要性（未排序）
    evaluator = ChannelImportanceEvaluator(base_model, device)
    print("Evaluating channel importance...")
    importance_dict = evaluator.evaluate(test_loader, threshold=0.5)
    print("Evaluation complete!")
    for name, scores in importance_dict.items():
        print(f"Layer {name}: Importance scores: {scores}")

    # 2. 收集待决策卷积层名称（仅考虑存在 importance 的层）
    conv_names = []
    for name, module in base_model.named_modules():
        if isinstance(module, nn.Conv2d) and name in importance_dict:
            conv_names.append(name)

    num_layers = len(conv_names)
    orig_channels = {layer: importance_dict[layer].shape[0] for layer in conv_names}

    # 3. 构造超结构网络（这里加入 GRU 与 BatchNorm）
    hsn = HyperStructureNetwork(num_layers=num_layers, d=16, gru_hidden_dim=16, h_dim=32, p=0.5, tau=1.0)
    hsn = hsn.to(device)

    # 4. 定义带动态 Mask 的模型，传入评估器得到的通道重要性
    masked_model = MaskedResNet(base_model, conv_names, hsn, importance_dict).to(device)

    # 用于记录各 Epoch 的统计信息
    retention_history = {layer: [] for layer in conv_names}
    retention_rate_history = {layer: [] for layer in conv_names}
    accuracy_history = []

    # 5. 超网络训练（目标：交叉熵损失 + 正则化损失）
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(list(hsn.parameters()) + list(masked_model.base_model.parameters()), lr=0.001)
    num_epochs = 30

    print("Starting hyper network training...")
    masked_model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs, reg_loss = masked_model(x)
            loss_cls = criterion(outputs, y)
            lam = 500
            if (epoch+2) % 7 == 0:
                loss = loss_cls + lam * reg_loss
            else:
                loss = loss_cls
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")
        if (epoch) % 7 == 0:
            # 重新评估通道重要性并生成全局 Mask
            hsn_outputs, _ = hsn()
            print("Hyper network outputs:", hsn_outputs)
            evaluator = ChannelImportanceEvaluator(base_model, device)
            print("Evaluating channel importance...")
            importance_dict = evaluator.evaluate(test_loader, threshold=0.5)
            final_importance_epoch = combine_importance(importance_dict, hsn_outputs, device, conv_names)

            binary_masks_epoch, global_threshold, sorted_positions, channels_info_sorted, total_param_val, target_keep = \
                global_binarize_by_param_budget(final_importance_epoch, base_model, target_param_keep_ratio=0.5, device=device)
            masked_model.bin_threshold = global_threshold

        # 统计当前被 mask为1 的通道总参数量
        retained_param_total = 0
        state_dict = base_model.state_dict()
        for layer in conv_names:
            weight_key = layer + ".weight"
            if weight_key not in state_dict:
                continue
            weight_tensor = state_dict[weight_key]
            bias_tensor = state_dict.get(layer + ".bias", None)
            mask_arr = binary_masks_epoch[layer].flatten()
            for i, m in enumerate(mask_arr):
                if m:
                    channel_cost = weight_tensor[i].numel()
                    if bias_tensor is not None:
                        channel_cost += 1
                    retained_param_total += channel_cost
        print(f"Epoch {epoch+1}: Retained channels total parameter count = {retained_param_total}")

        # 记录每层保留通道数和保留率
        for layer in conv_names:
            mask = binary_masks_epoch[layer]
            retention_count = int(np.sum(mask))
            retention_history[layer].append(retention_count)
            orig = orig_channels[layer]
            rate = retention_count / orig * 100.0
            retention_rate_history[layer].append(rate)
            print(f"Epoch {epoch+1}, Layer {layer}: retained channels = {retention_count}, retention rate = {rate:.2f}%")

        # 测试阶段计算精度
        masked_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs, _ = masked_model(images)
                _, predicted = torch.max(outputs, dim=1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        test_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}: Test Accuracy = {test_acc:.2f}%")
        accuracy_history.append(test_acc)
        masked_model.train()

    print("Hyper network training complete!")

    # 绘制统计图
    epochs = np.arange(1, num_epochs+1)
    fig, axes = plt.subplots(3, 1, figsize=(10, 14), sharex=True)
    for layer in conv_names:
        axes[0].plot(epochs, retention_history[layer], marker='o', label=f"{layer}")
    axes[0].set_ylabel("Retained Channel Count")
    axes[0].set_title("Retention Channel Count per Layer Over Epochs")
    axes[0].legend()
    axes[0].grid(True)
    for layer in conv_names:
        axes[1].plot(epochs, retention_rate_history[layer], marker='s', label=f"{layer}")
    axes[1].set_ylabel("Retention Rate (%)")
    axes[1].set_title("Retention Rate per Layer Over Epochs")
    axes[1].legend()
    axes[1].grid(True)
    axes[2].plot(epochs, accuracy_history, marker='d', color='tab:purple', label="Test Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Test Accuracy (%)")
    axes[2].set_title("Test Accuracy Over Epochs")
    axes[2].legend()
    axes[2].grid(True)
    plt.tight_layout()
    plt.savefig("img/retention_over_epochs.png")
    plt.show()

    # 最后进行一次全局重要性重排，打印部分全局重要性信息
    hsn_outputs, _ = hsn()
    final_importance = combine_importance(importance_dict, hsn_outputs, device, conv_names)
    print("Final combined channel importance (partial):")
    first_layer = conv_names[0]
    print(f"Layer {first_layer}: {final_importance[first_layer]}")
    print("Final channel reordering based on global importance (cross-layer)...")
