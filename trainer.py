
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
from channel_evaluation import ChannelImportanceEvaluator,DataLoader,combine_importance,HyperStructureNetwork,MaskedResNet
class FedPRC_trainer(basic.Trainer):
    def __init__(self, model=None, callbacks=None):
        super().__init__(model=model, callbacks=callbacks)
        # self.model = model(**Config().parameters.model._asdict())
        self.current_loss = None
    def train_step_end(self, config, batch=None, loss=None):
        # self.current_loss = loss.clone()
        
        return super().train_step_end(config, batch, loss)


    

    
    

