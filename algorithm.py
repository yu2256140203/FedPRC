"""
FedRolexfl algorithm.
"""

import sys
import pickle
import random
import copy

import torch
import numpy as np
import ptflops
from plato.config import Config
from plato.algorithms import fedavg
from prune import prune_state_dict,reverse_prune_rates,get_channel_indices,aggregate_submodel_states

class FedPRC_algorithm(fedavg.Algorithm):
    """A federated learning algorithm using the FedRolexfl algorithm."""

    def __init__(self, trainer=None):
        super().__init__(trainer)
        self.current_rate = 1
        self.model_class = None
        # self.rates = [1.0, 0.5, 0.25, 0.125, 0.0625]
        self.mapping_indices = None

    # def extract_weights(self, model=None):
    #     self.model = self.model.cpu()
    #     payload = self.model.state_dict()
    #     return payload
    #分割子模型
    def sub_weights(self,payload,mapping_indices):
        sub_weights = self.get_local_parameters(weight=payload,mapping_indices=mapping_indices)
        
        # print(sub_weights["layer1.0.conv1.weight"].shape)
        # print(self.mapping_indices)
        # print(self.model.state_dict().keys())
        # self.model.load_state_dict(sub_weights)
        return sub_weights




    def get_local_parameters(self,weight,mapping_indices):
        """
        Get the parameters of local models from the global model.
        """

        local_parameters = prune_state_dict(weight, mapping_indices)


        return local_parameters
    def get_test_parameters(self,model,mapping_dict):
        """
        Get the parameters of local models from the global model.
        """
        local_parameters_list = {}
        for key,value in mapping_dict.items():
            local_parameters_list[key] = prune_state_dict(model.state_dict(), value)
        



        return local_parameters_list

    # def aggregation(self, weights_received):
    #     #有它，聚合参数，否则聚合变量
    #     """
    #     Aggregate weights of different complexities.
    #     """

    #     submodel_weights = []
    #     mapping_indices_list = []
    #     client_data_sizes = []
    #     for payload in weights_received:
    #         submodel_weights.append(payload["outbound_payload"])
    #         mapping_indices_list.append(payload["mapping_indices"])
    #         client_data_sizes.append(payload["data_size"])
            
    #     global_parameters,restored_states = aggregate_submodel_states(full_state=self.model.state_dict(),sub_state_list=submodel_weights,mapping_indices_list=mapping_indices_list,client_data_sizes=client_data_sizes)
        
    #     return global_parameters,restored_states
    def aggregation(self, weights_received):
        #有它，聚合参数，否则聚合变量
        """
        Aggregate weights of different complexities.
        """
        client_weights = []
        client_datas    = []
        for client_weight,client_data in  weights_received:
            client_weights.append(client_weight)
            client_datas.append(client_data)

        aggregated_state = {}
        total_data = sum(client_datas)
        full_state = self.model.state_dict()
        for key in full_state.keys():
            # 使用各客户端的数据量作为权重进行加权平均
            agg_param = sum((client_datas[i] / total_data) * client_weights[i][key] 
                            for i in range(len(client_weights)))
            aggregated_state[key] = agg_param

        return aggregated_state
    
    def load_weights(self, weights):
        """Loads the model weights passed in as a parameter."""
        self.model.load_state_dict(weights, strict=False)#非严格性参数加载，避免有些参数有点多

