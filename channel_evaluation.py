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


class ChannelImportanceEvaluator:
    def __init__(self, model, device):
        self.model = model.to(device)
        self.device = device

        # 用于存放卷积层的累计评分
        # { layer_name: { "A_sum": Tensor(C), "B_sum": Tensor(C), "count": int } }
        self.score_dict_conv = {}

        # self.threshold 用于卷积层的二值化阈值
        self.threshold = None
        # 当前批次输入的原图和二值化后的 mask，用于卷积 A_score 计算
        self.current_orig = None
        self.current_binary = None

        if Config().data.datasource in ["CIFAR10", "CIFAR100"]:
            self.target_size = (32, 32)
        else:
            self.target_size = (64, 64)

    def hook_conv(self, name):
        """
        针对每个 nn.Conv2d 层的 forward hook。
        计算 A_score（基于二值化后的 IoU）和 B_score（基于输出范数 / 上一层方差），
        并把它们累加到 self.score_dict_conv[name] 中。
        """
        def _hook(module, inputs, output):
            # output: Tensor, 形状 (B, C, H, W)
            # 1) 先插值到 target_size，然后二值化
            up_out = F.interpolate(output, size=self.target_size, mode="bilinear", align_corners=False)
            binary_feat = (up_out > self.threshold).float()  # (B, C, H', W')
            binary_orig = self.current_binary.expand_as(binary_feat)  # (B, C, H', W')

            B, C, H, W = binary_feat.shape
            # 2) 计算交集与并集 (IoU)——对每个通道分别算 IoU
            inter = (binary_feat * binary_orig).view(B, C, -1).sum(dim=2)      # (B, C)
            union = (((binary_feat + binary_orig) > 0).float()).view(B, C, -1).sum(dim=2)  # (B, C)
            iou = inter / (union + 1e-6)  # (B, C)
            A_score = iou.mean(dim=0)     # (C,)

            # 3) 计算 B_score：输出的范数 / 上一层输入的方差
            norm_sq = output.pow(2).sum(dim=(2, 3))  # (B, C)
            var_orig = self.current_orig.var(dim=(2, 3), unbiased=False).mean(dim=1, keepdim=True)  # (B, 1)
            B_score = (norm_sq / (var_orig + 1e-6)).mean(dim=0)  # (C,)

            # 4) 把 A_score 和 B_score 累加到字典里
            if name not in self.score_dict_conv:
                self.score_dict_conv[name] = {
                    "A_sum": A_score.clone(),
                    "B_sum": B_score.clone(),
                    "count": 1
                }
            else:
                self.score_dict_conv[name]["A_sum"] += A_score
                self.score_dict_conv[name]["B_sum"] += B_score
                self.score_dict_conv[name]["count"] += 1

        return _hook

    def evaluate(self, data_loader, threshold=0.5):
        """
        对给定的 data_loader，遍历所有样本，收集每个卷积层的输出统计量；
        同时在最后一步遍历完毕后，对所有线性层直接根据权重计算 L2 范数 + Z‐Score。
        返回一个字典：{ layer_name: importance_numpy_array }。
        """
        self.threshold = threshold
        hooks = []

        # 1. 为卷积层注册 forward hook
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Conv2d):
                h = module.register_forward_hook(self.hook_conv(name))
                hooks.append(h)

        self.model.eval()
        with torch.no_grad():
            for x, _ in data_loader:
                x = x.to(self.device)
                # 保存原始输入，用于后续计算方差
                self.current_orig = x.clone()
                # 生成当前批次的二值化 mask，用于 A_score 计算
                gray = x.mean(dim=1, keepdim=True)  # (B, 1, H, W)
                self.current_binary = F.interpolate(gray, size=self.target_size, mode="bilinear", align_corners=False)
                self.current_binary = (self.current_binary > threshold).float()  # (B, 1, H', W')

                # 只做前向传播，让所有 Conv2d 的 hook 执行
                _ = self.model(x)

        # 卷积层 hook 移除
        for h in hooks:
            h.remove()

        importance_dict = {}

        # 2. 处理卷积层的累计统计 → 最终 importance 
        #    先对 A_sum / count, B_sum / count 做 gamma 变换和 Z‐Score，再相乘，最后再做一轮 Z‐Score
        for name, scores in self.score_dict_conv.items():
            count = scores["count"]
            A_avg = scores["A_sum"] / count       # Tensor, 形状 (C,)
            B_avg = scores["B_sum"] / count       # Tensor, 形状 (C,)
            gamma = 0.2

            # # 做 γ 幂运算
            A_transformed = torch.pow(A_avg, gamma)
            B_transformed = torch.pow(B_avg, gamma)

            # # 对 A_transformed 做首轮 Z‐Score
            mean_A = A_transformed.mean()
            std_A = A_transformed.std() + 1e-6
            norm_A = (A_transformed - mean_A) / std_A

            # # 对 B_transformed 做首轮 Z‐Score
            mean_B = B_transformed.mean()
            std_B = B_transformed.std() + 1e-6
            norm_B = (B_transformed - mean_B) / std_B
            
            # # 乘起来作为“综合重要性”
            importance_raw = norm_A * norm_B  # Tensor, 形状 (C,)
            # importance_raw = A_avg * B_avg

            # 再做一次 Z‐Score，使每个层输出的通道分数均值为 0，方差为 1
            mean_imp = importance_raw.mean()
            std_imp = importance_raw.std() + 1e-6
            z_score_importance = (importance_raw - mean_imp) / std_imp  # Tensor, (C,)

            importance_dict[name] = z_score_importance.detach().cpu().numpy()

        # 3. 处理线性层：直接用 weight 的 L2 范数 + Z‐Score，生成 importance
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                # module.weight 形状 (out_features, in_features)
                with torch.no_grad():
                    W = module.weight.data  # Tensor, (out_features, in_features)
                    # 计算每个输出神经元（第 j 行）的 L2 范数
                    # -> shape (out_features,)
                    l2_norm = W.norm(p=2, dim=1)

                    # 做 Z‐Score 标准化
                    mean_w = l2_norm.mean()
                    std_w = l2_norm.std() + 1e-6
                    z_score_w = (l2_norm - mean_w) / std_w  # Tensor, (out_features,)

                    importance_dict[name] = z_score_w.detach().cpu().numpy()

        # 保证 CUDA 异步计算完成
        torch.cuda.synchronize(self.device)
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
        # print("当前对应关系:",name,hsn_val)
        # 将结果直接存入字典（不调用 detach、.cpu() 或 .numpy()）
        final_importance[name] = final_imp
        # def print_grad(grad):
        #     print("hsn_outputs gradient:", grad)

        # # 为 hsn_outputs[i] 注册 hook
        # handle = hsn_outputs[i].register_hook(print_grad)


    return final_importance


class MaskedResNet(nn.Module):
    def __init__(self, base_model, hsn, important,probab_masks):
        super().__init__()
        self.base_model = base_model
        self.probab_masks = probab_masks
        self.hsn = hsn
        self.conv_names = self.probab_masks.keys()
        self.important = important

    def forward(self, x, update=False):
        device = next(self.base_model.parameters()).device
        self.hsn_outputs,reg_loss = self.hsn()
        final_importance = combine_importance(importance_dict=self.important, hsn_outputs=self.hsn_outputs, device=device, conv_names=self.conv_names)
        self.probab_masks  = {}  # 初始化概率掩码字典
        T = 1.0  
        tau = 0.5 # 阈值（可调整）
        for layer_name, imp in final_importance.items():
            if isinstance(imp, torch.Tensor):
                imp_tensor = imp.to(device)
            else:
                imp_tensor = torch.tensor(imp, device=device, dtype=torch.float)
            self.probab_masks[layer_name] = torch.sigmoid(imp_tensor * T - tau)

        def hook_fn(module, inputs, output, name):
            if output.ndim == 4:  # 表示形状是 (50, 512, 2, 2)
                mask_expanded = self.probab_masks[name].reshape(1, -1, 1, 1).clone()
            elif output.ndim == 2:  # 表示形状是 (50, 512)
                mask_expanded = self.probab_masks[name].reshape(1, -1).clone()
            else:
                raise ValueError(f"Unexpected mask shape: {output.shape}")
            # mask_expanded = self.probab_masks[name].reshape(1, -1, 1, 1).clone()
            
            # print("output维度",output.shape)
            # print("mask维度",mask_expanded.shape)
            output = output * mask_expanded
            # print("output相乘后于",output.shape)

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




