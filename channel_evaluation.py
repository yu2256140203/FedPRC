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
from plato.config import Config

# 如果 img 文件夹不存在则创建
if not os.path.exists("img"):
    os.makedirs("img")



# 通道重要性评估模块 (使用 forward hook)

class ChannelImportanceEvaluator:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device
        self.score_dict = {}
        self.threshold = None
        self.current_orig = None
        self.current_binary = None
        if Config().data.datasource == "CIFAR10" or Config().data.datasource == "CIFAR100":
                self.target_size = (32, 32)
        else:
                self.target_size = (64, 64)

    def hook_fn(self, name):
        def _hook(module, inputs, output):
            

            up_out = F.interpolate(output, size=self.target_size, mode="bilinear", align_corners=False)
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
                self.current_binary = F.interpolate(gray, size=self.target_size, mode="bilinear", align_corners=False)
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



#  超结构网络 (HyperStructure Network, HSN) —— 加入 GRU 与 BatchNorm

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
        # if update:
        #     return out, reg_loss
        return out




