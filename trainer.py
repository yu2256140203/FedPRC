
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
                # 统计准确率
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                print("目前有",correct,"一样")
        accuracy =  correct / total
        return accuracy


    

    
    

