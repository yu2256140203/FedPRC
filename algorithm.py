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
        print(mapping_indices["conv1.weight"])
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

    def aggregation(self, weights_received):
        #有它，聚合参数，否则聚合变量
        """
        Aggregate weights of different complexities.
        """

        submodel_weights = []
        mapping_indices_list = []
        for payload in weights_received:
            submodel_weights.append(payload["outbound_payload"])
            mapping_indices_list.append(payload["mapping_indices"])
        global_parameters,restored_states = aggregate_submodel_states(full_state=self.model.state_dict(),sub_state_list=submodel_weights,mapping_indices_list=mapping_indices_list)
        
        return global_parameters,restored_states

