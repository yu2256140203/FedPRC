
from plato.config import Config
from plato.samplers import registry as samplers_registry
from plato.trainers import registry as trainers_registry
from plato.algorithms import registry as algorithms_registry
from plato.datasources import registry as datasources_registry
from plato.processors import registry as processor_registry
from plato.clients import simple
import logging
from plato.utils import fonts
from types import SimpleNamespace
import time
import os
import torch
from channel_evaluation import combine_importance,HyperStructureNetwork,ChannelImportanceEvaluator,MaskedResNet,global_binarize_by_param_budget
import torch.nn as nn
from torch.utils.data import DataLoader
from prune import prune_state_dict,reverse_prune_rates,get_channel_indices
class client(simple.Client):
    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
        callbacks=None,
        trainer_callbacks=None,
        rate=None
    ):
        #Yu 这里一定要好好写啊，不然client不知道它有啥，还是按照默认的来
        super().__init__(callbacks=callbacks,algorithm=algorithm,model=model,trainer=trainer)
        #为每个客户端分配不同的参数百分比
        self.rate = rate
        self.local_global_ranks = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

         # 初始化通道评估器：用于后续在线评价通道重要性
        self.base_model = self.model
        # 收集所有卷积层名称（后续 eval_tunnels 中会依 importance_dict 筛选）
        self.conv_names = []
        self.output_dims = []
        self.bin_threshold = None
        self.prune_rates = None
        self.mapping_indices = None

        #下面是用于通道评估的各种组件
        self.test_loader = None
        self.channel_importance_dict = None

        

    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    def process_server_response(self, server_response) -> None:
        self.rate = server_response["rate"]
        print("当前客户端剪枝率",self.rate * self.rate)
        self.mapping_indices = server_response["mapping_indices"]
        self.prune_rates = reverse_prune_rates(self.mapping_indices, Config().parameters.hidden_size)
        # print(self.mapping_indices["layer1.0.conv1.weight"])
        print(self.prune_rates)

        self.algorithm.model = self.model(
           self.prune_rates
        )
        print("当前客户端:",self.client_id,"参数量：",self.count_parameters(self.algorithm.model))

        self.algorithm.mapping_indices = self.mapping_indices
        self.trainer.model = self.algorithm.model
        self.channel_importance_dict = server_response["channel_importance_dict"]
    # #将全局模型切割为子模型
    # def _load_payload(self, server_payload) -> None:
    #     """Loads the server model onto this client."""
    #     self.algorithm.sub_weights(server_payload)

    async def _start_training(self, inbound_payload):
        """Complete one round of training on this client."""
        self._load_payload(inbound_payload)
        report, outbound_payload = await self._train()
        outbound_payload = {"acc":report.accuracy ,"outbound_payload":outbound_payload,"mapping_indices":self.mapping_indices,"client_id":self.client_id}
        print(report.accuracy)
        if Config().is_edge_server():
            logging.info(
                "[Server #%d] Model aggregated on edge server (%s).", os.getpid(), self
            )
        else:
            logging.info("[%s] Model trained.", self)
        

        #对各种评价工具进行初始化
        self.train_loader = self.trainer.get_train_loader(trainset=self.trainset,sampler=self.sampler.get(),batch_size=32,shuffle=False)

        # if  self.channel_importance_dict == None or (self.current_round % Config().parameters.ranks_round == 0 and self.current_round != 0):

        # if   (self.current_round % Config().parameters.ranks_round == 0 or  self.current_round == 1):
        if   (self.current_round % Config().parameters.ranks_round == 0) or self.current_round==1:


        
            #如果当前的重要性为空的话，直接返回重要性，尽快的下次下发正确的模型，然后再待20轮后再更新当前重要性
            self.evaluator = ChannelImportanceEvaluator(self.trainer.model, self.device)
            self.channel_importance_dict = self.evaluator.evaluate(self.train_loader, threshold=0.5)
            # print(self.channel_importance_dict)
            print("Evaluation complete!")
            outbound_payload["channel_importance_dict"] = self.channel_importance_dict
            # outbound_payload["loss"] = self.trainer.current_loss
            # print(self.trainer.current_loss)
            # import time
            # time.sleep(60)

        else:
            outbound_payload["channel_importance_dict"] = None
            #节省通信量
        outbound_payload["loss"] = None

        return report, outbound_payload
    # def customize_report(self, report):
    #     self.acc = report.accuracy
    #     return super().customize_report(report)
    
        





