
from plato.trainers import mindspore,basic
from plato.trainers.basic import models_registry

from plato.config   import Config
import os
import torch
import pickle
import logging
import multiprocessing as mp
import torch.nn as nn
import time
import copy
from torch.utils.data import DataLoader
class FedPRC_trainer(basic.Trainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(model=model, callbacks=callbacks)
        # self.model = model(**Config().parameters.model._asdict())
        self.current_loss = None
    def train_step_end(self, config, batch=None, loss=None):
        # self.current_loss = loss.clone()
        
        return super().train_step_end(config, batch, loss)
    def custom_server_test(self,test_set):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        testloader = DataLoader(test_set, batch_size=Config().trainer.batch_size, shuffle=False)
        model = self.model.to(device)
        model.eval()
        # 在test_server_accuracy函数中添加
        def check_model_weights(model):
            """检查模型权重的分布情况"""
            print("\n===== 模型权重分析 =====")
            for name, param in model.named_parameters():
                if param.requires_grad:
                    print(f"{name}: min={param.data.min().item():.6f}, max={param.data.max().item():.6f}, "
                        f"mean={param.data.mean().item():.6f}, std={param.data.std().item():.6f}")
                    
                    # 检查是否有NaN或Inf
                    if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                        print(f"警告: {name} 包含NaN或Inf值!")
                        
                    # 检查是否所有值都相同
                    if param.data.min() == param.data.max():
                        print(f"警告: {name} 所有值都相同!")
        def check_bn_layers(model):
            """检查模型中批归一化层的状态"""
            print("\n===== 批归一化层分析 =====")
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d):
                    print(f"{name}:")
                    print(f"  running_mean: min={module.running_mean.min().item():.4f}, "
                        f"max={module.running_mean.max().item():.4f}, "
                        f"mean={module.running_mean.mean().item():.4f}")
                    print(f"  running_var: min={module.running_var.min().item():.4f}, "
                        f"max={module.running_var.max().item():.4f}, "
                        f"mean={module.running_var.mean().item():.4f}")
                    print(f"  weight: {module.weight.data}")
                    print(f"  bias: {module.bias.data}")
        def check_pruned_channels(model):
            """检查剪枝后每层的有效通道数"""
            print("\n===== 剪枝后通道分析 =====")
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    # 检查权重是否有全零行/列
                    weight = module.weight.data
                    
                    # 检查输出通道是否全为零
                    out_channels_sum = weight.sum(dim=(1, 2, 3))
                    zero_out_channels = (out_channels_sum == 0).sum().item()
                    
                    # 检查输入通道是否全为零
                    in_channels_sum = weight.sum(dim=(0, 2, 3))
                    zero_in_channels = (in_channels_sum == 0).sum().item()
                    
                    print(f"{name}: 输出通道={weight.size(0)}, 全零输出通道={zero_out_channels}, "
                        f"输入通道={weight.size(1)}, 全零输入通道={zero_in_channels}")





        # 添加模型诊断
        # check_model_weights(model)
        # check_bn_layers(model)
        # check_pruned_channels(model)

# 然后继续测试...

        # 测试模型
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)               
                # 获取预测结果
                _, predicted = torch.max(outputs.data, 1)
                # print(outputs[0])
                # print(predicted[0]) 
                # 统计准确率
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # print("目前有",correct,"一样")
        accuracy =  correct / total
        return accuracy


    

    
    

