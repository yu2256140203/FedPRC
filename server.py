"""
HeteroFL algorithm trainer.
"""
import copy
import numpy as np

from plato.config import Config
from plato.servers import fedavg
from plato.samplers import all_inclusive
import logging
from plato.utils import s3, fonts
import sys
import pickle
import random
import torch
import os
from prune import get_channel_indices_unuse_hsn,client_state_dict,prune_state_dict,reverse_prune_rates,get_channel_indices,aggregate_submodel_states,prune_mapping_with_global_threshold_and_binary_indices
from channel_evaluation import HyperStructureNetwork,combine_importance,global_binarize_by_param_budget,MaskedResNet
from torch.utils.data import Subset
class server(fedavg.Server):
    """Federated Learning - Pruning for Resource-Constrained."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
    ):
        # pylint:disable=too-many-arguments
        super().__init__(model, datasource, algorithm, trainer)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.server_importance_dicts = {name: [0.0] * module.out_channels for name, module in model().named_modules() 
        if 'conv' in name} #初始化每层的通道重要性
        #我并不需要在全局直接算出来当前的full_model_global_ranks，毕竟按照通道重要性合并完，再计算也一样

        
        self.channel_importance_dicts = [None for _ in range(Config().clients.total_clients)]#本地当前轮次全局重要性排名
        self.client_losses = []
        self.client_full_model_global_ranks = [None for _ in range(Config().clients.total_clients)]
        #客户端的每个通道对于完整模型的重要性排名
        self.clients_mapping_indices = [None for _ in range(Config().clients.total_clients)]
        self.probab_masks = [None for _ in range(Config().clients.total_clients)]

        # self.current_linked_client_ids = None # 当前轮次的客户端id，但是有序的，按照通信顺序来的
        self.rates = [0 for _ in range(Config().clients.total_clients)]
        self.init_mapping = get_channel_indices(model(),Config().parameters.hidden_size)
        
        # 获取配置中的参数
        self.rates_values = Config().parameters.rates
        parameters_to_clients_ratio = Config().parameters.Parameters_to_percent_of_clients
        self.prune_rates = [None for _ in range(Config().clients.total_clients)]
        
        # 根据 Parameters_to_percent_of_clients 的比例分配客户端
        total_clients = Config().clients.total_clients
        clients_per_group = total_clients // len(parameters_to_clients_ratio)
        self.deltas = None
        #随机的三种mapping
        self.random_mapping = []

        for client_idx in range(total_clients):
            if client_idx < (total_clients * parameters_to_clients_ratio[0]) // sum(parameters_to_clients_ratio):
                group_index = 0
            elif client_idx < total_clients * (parameters_to_clients_ratio[0]+ parameters_to_clients_ratio[1]) // sum(parameters_to_clients_ratio):
                group_index = 1
            else:
                group_index = 2
            # 设置 rates
            self.rates[client_idx] = self.rates_values[group_index]
        for rate in self.rates_values:
        #     prune_rates = [
        #     [[rate, rate], [rate, rate]],  # layer1
        #     [[rate, rate], [rate, rate]],  # layer2
        #     [[rate, rate], [rate, rate]],  # layer3
        #     [[rate, rate], [rate, rate]]   # layer4
        # ]
            self.random_mapping.append(get_channel_indices(self.model(),Config().parameters.hidden_size, rate))



        if Config().parameters.FlexFL_exp:
            self.rates = [0 for _ in range(Config().clients.total_clients)]
            self.clients_FLexFL_exp  = self.summon_clients(total_clients)


            



        self.train_model = None
        # self.conv_names = [i for i in model().state_dict().keys() if 'conv' in i and i != Config().parameters.first_layer]
        self.conv_names = [name for name, module in model().named_modules()  if 'conv' in name and name != Config().parameters.first_layer] #初始化每层的通道重要性
        num_layers = len(self.conv_names)
        # self.hsn = HyperStructureNetwork(num_layers=num_layers, d=16, gru_hidden_dim=16, h_dim=32, p=0.5, tau=1.0).cuda()
        self.hsn = HyperStructureNetwork(num_layers=num_layers, d=16, gru_hidden_dim=16, h_dim=32, p=0.5).cuda()

        self.current_hsn_output,self.reg_loss = self.hsn()
        # self.hyper_optimizer  = torch.optim.SGD(self.hsn.parameters(), lr=0.01,momentum=0.9)
        self.hyper_optimizer  = torch.optim.SGD(self.hsn.parameters(), lr=0.1)

        for i in range(30):
            self.hyper_optimizer.zero_grad()
            self.reg_loss.backward(retain_graph=True)  # 触发反向传播
            self.hyper_optimizer.step()
            #初始的超网络输出：
            self.current_hsn_output,self.reg_loss = self.hsn()
            print(self.current_hsn_output)
        self.hyper_optimizer  = torch.optim.SGD(self.hsn.parameters(), lr=0.1,momentum=0.9)
        #重新定义超网络优化器，探索重要性
        #希望初始超网络的输出能公平少一点随机，这样让初始的模型能有较好的性能
        self.acc = [0.0 for _ in range(Config().clients.total_clients)]#本地当前轮次全局重要性排名
        self.proxy_data_trainloader = None
        self.proxy_data_testloader = None
    def init_proxy(self):
        dataset_train, dataset_test = self.datasource.trainset, self.datasource.testset
        # Split 80% test dataset to proxy train dataset, 20% test dataset to proxy test dataset for getting APoZ
        total_size = len(dataset_test)
        indices = list(range(total_size))
        split = int(np.floor(Config().parameters.proxy_data_rate * total_size))
        np.random.shuffle(indices)
        sub_train_index, sub_test_index = indices[:split], indices[split:]
        sub_train_dataset = Subset(dataset_test, sub_train_index)
        sub_test_dataset = Subset(dataset_test, sub_test_index)
        self.proxy_data_trainloader = torch.utils.data.DataLoader(sub_train_dataset, batch_size=Config().trainer.batch_size, shuffle=True)
        self.proxy_data_testloader  = torch.utils.data.DataLoader(sub_test_dataset, batch_size=Config().trainer.batch_size, shuffle=True)
    
        

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        server_response["rate"] =  self.rates[client_id - 1]
        rate = server_response["rate"]
        if Config().parameters.FlexFL_exp:
            resource = self.clients_FLexFL_exp[client_id-1][0] - abs(np.random.normal(0, self.clients_FLexFL_exp[client_id-1][1], 1)[0])
            server_response["rate"] = self.resource_to_model3(resource)
            rate = server_response["rate"]
        # server_response["client_full_model_global_ranks"] = self.client_full_model_global_ranks[client_id-1]
        
        #因为客户端做的是模拟，需要每次换新变量，不然就读取本地的形式
        if self.channel_importance_dicts[client_id-1] != None:
            server_response["channel_importance_dict"] = self.channel_importance_dicts[client_id-1]
        else:
            server_response["channel_importance_dict"] = None


        
        if self.current_round == 1:
            self.init_proxy()#初始化代理数据集
            #每轮随机
            # self.random_mapping = []
            # for rate1 in self.rates_values:
            #     self.random_mapping.append(get_channel_indices(self.model(),Config().parameters.hidden_size, rate1))
            
            
            self.clients_mapping_indices[self.selected_client_id-1] = self.random_mapping[self.rates_values.index(rate)]
            server_response["mapping_indices"] = self.clients_mapping_indices[self.selected_client_id-1]
            # self.clients_mapping_indices[self.selected_client_id-1] = server_response["mapping_indices"]
        elif Config().parameters.unuse_hsn != None and Config().parameters.unuse_hsn == True:
            self.clients_mapping_indices[self.selected_client_id-1] = get_channel_indices_unuse_hsn(model=self.model(),submodel_layer_prune_rate=rate,importance=self.server_importance_dicts)
            server_response["mapping_indices"] = self.clients_mapping_indices[self.selected_client_id-1]
            
        else:
            client_full_model_global_rank = combine_importance(importance_dict=self.server_importance_dicts, hsn_outputs=self.current_hsn_output, device=self.device, conv_names=self.conv_names)
            # binary_indices, global_threshold, sorted_positions, channels_info_sorted, total_param_val, target_keep = \
            #     global_binarize_by_param_budget(client_full_model_global_rank, self.model(), target_param_keep_ratio=rate, device=self.device)
            # masked_model.bin_threshold = global_threshold
            # print(binary_indices)
            # print("当前正在打印")
            
            pruned_mapping,binary_masks, probab_masks, binary_indices, channels_info_sorted, total_param, target_keep  =prune_mapping_with_global_threshold_and_binary_indices(initial_mapping=self.init_mapping,
            final_importance=client_full_model_global_rank, model=self.model(),target_ratio=rate * rate, device=self.device)
            self.clients_mapping_indices[self.selected_client_id-1 ] = pruned_mapping
            self.probab_masks[self.selected_client_id-1 ] = probab_masks
            server_response["mapping_indices"] = self.clients_mapping_indices[self.selected_client_id-1 ]
            

        

        return super().customize_server_response(server_response, client_id)
    
    async def aggregate_weights(
        self, updates, baseline_weights, weights_received
    ):  # pylint: disable=unused-argument
        # 更新当前获得的重要性
        for payload in weights_received:
            self.acc[payload["client_id"] - 1] = payload["acc"]
        if weights_received[0]["channel_importance_dict"] is not None:
            for payload in weights_received:
                for key,value in payload["channel_importance_dict"].items():
                    payload["channel_importance_dict"][key] = payload["channel_importance_dict"][key].tolist()
                self.channel_importance_dicts[payload["client_id"] - 1] = payload["channel_importance_dict"]
                
                
                    
            # 对每层进行处理
            for key in self.server_importance_dicts.keys():
                # 初始化累加张量和计数张量
                agg_tensor = torch.zeros(len(self.server_importance_dicts[key]), device=self.device)
                count_tensor = torch.zeros(len(self.server_importance_dicts[key]), device=self.device)
                
                # 聚合每个客户端的重要性到服务端张量中
                for payload in weights_received:
                    # print(payload["channel_importance_dict"].keys())
                    #只使用卷积权重的通道索引即可
                    for idx, importance_value in zip(self.clients_mapping_indices[payload["client_id"] - 1][key+".weight"][-1], 
                                                    payload["channel_importance_dict"][key]):
                        agg_tensor[idx] += importance_value
                        count_tensor[idx] += 1
                
                # 对累加结果进行平均，并处理未累加通道
                nonzero = count_tensor > 0  # 标记有累加值的位置
                avg_tensor = torch.zeros_like(agg_tensor)
                avg_tensor[nonzero] = agg_tensor[nonzero] / count_tensor[nonzero]
                # print(len(avg_tensor),len(self.server_importance_dicts[key]))
                
                # 对于未累加的通道，使用原来的 self.server_importance_dicts 的值
                # 获取 avg_tensor 中未累加的位置索引
                missing_indices = torch.where(~nonzero)[0]  # torch.where 会返回未累加值的索引列表

                # 通过这些索引获取 self.server_importance_dicts[key] 的对应值
                server_values_for_missing = torch.tensor(self.server_importance_dicts[key], device=self.device)[missing_indices]

                # 更新 avg_tensor 的未累加位置，使用来自 server_importance_dicts 的值
                avg_tensor[missing_indices] = server_values_for_missing

                
                # 更新 server_importance_dicts 的值为计算后的 avg_tensor
                self.server_importance_dicts[key] = avg_tensor
                print("当前完成重要性聚合")
        print("当前完成聚合")
        
        last_global_parameters = copy.deepcopy(self.algorithm.extract_weights())
        """Aggregates weights of models with different architectures."""
        global_parameters,restored_models = self.algorithm.aggregation(weights_received)
        #获取更新超网络的deltas
        #yu 当前所有的客户端的mask应当是一致的，所以可以只调用一次超网络反向传播
        # if  Config().parameters.unuse_hsn != None and  Config().parameters.unuse_hsn != True and  self.current_round % Config().parameters.ranks_round == 0 and self.current_round != Config().parameters.ranks_round:
        if  Config().parameters.unuse_hsn != None and  Config().parameters.unuse_hsn != True and  self.current_round % Config().parameters.ranks_round == 0 and self.current_round != 1:

            self.deltas  = None
            for idx, (client_parameters, payload) in enumerate(zip(restored_models, weights_received)):
                # 强行转化成列表
                self.deltas = copy.deepcopy(self.algorithm.compute_weight_deltas(last_global_parameters, [client_parameters]))
                
                # if self.probab_masks[payload["client_id"] - 1] is not None and idx == len(weights_received) - 1:
                #     #如果是最后一项，释放掉计算图
                #     self.update_hsn(self.deltas[0], self.probab_masks[payload["client_id"] - 1],save_graph=False)
                # elif self.probab_masks[payload["client_id"] - 1] is not None:
                #     self.update_hsn(self.deltas[0], self.probab_masks[payload["client_id"] - 1])
                self.update_hsn(self.deltas[0], self.probab_masks[payload["client_id"] - 1])
                break

                    
        
        return global_parameters

    def get_logged_items(self) -> dict:
            """Get items to be logged by the LogProgressCallback class in a .csv file."""
            logged_items = super().get_logged_items()
            #round, accuracy, avg_accuracy, large_accuracy, middle_accuracy, small_accuracy
            self.avg_acc = list()
            self.large_acc =list()
            self.middle_acc = list()
            self.small_acc = list()
            # print(self.selected_clients)
            for i in self.selected_clients:
                self.avg_acc.append(self.acc[i-1])
                x = self.rates_values.index(self.rates[i-1])
                if x == 0:
                    self.small_acc.append(self.acc[i-1])
                elif x == 1:
                    self.middle_acc.append(self.acc[i-1])
                elif x == 2:
                    self.large_acc.append(self.acc[i-1])
            # print(self.avg_acc,self.large_acc,self.middle_acc,self.small_acc)
            if len(self.avg_acc) == 0:
                avg_acc = 0.0
            else:
                # print(sum(self.avg_acc), len(self.avg_acc))
                avg_acc = sum(self.avg_acc) / len(self.avg_acc)

            if len(self.large_acc) == 0:
                large_acc = 0.0
            else:
                large_acc = sum(self.large_acc) / len(self.large_acc)

            if len(self.middle_acc) == 0:
                middle_acc = 0.0
            else:
                middle_acc = sum(self.middle_acc) / len(self.middle_acc)

            if len(self.small_acc) == 0:
                small_acc = 0.0
            else:
                small_acc = sum(self.small_acc) / len(self.small_acc)
                    

            # clusters_accuracy = "; ".join([str(acc) for acc in clusters_accuracy])

            logged_items["avg_accuracy"] = avg_acc
            logged_items["large_accuracy"] = large_acc
            logged_items["middle_accuracy"] = middle_acc
            logged_items["small_accuracy"] = small_acc

            return logged_items
    
    def customize_server_payload(self, payload):
        #定制子模型
        payload = self.algorithm.sub_weights(payload,self.clients_mapping_indices[self.selected_client_id-1])      
        return super().customize_server_payload(payload)
    

    # def update_hsn(self, deltas):
    #     """
    #     利用客户端更新的 deltas 构造代理损失，使得对模型参数的梯度正好等于 deltas，
    #     然后利用整个计算图将该梯度传递回超网络。

    #     注意：这里对 deltas 使用 detach()，把它当作常量。
    #     """
    #     deltas = deltas[0]
    #     # 初始值不重要，但要确保后续加入的项参与梯度传递。
    #     surrogate_loss = None
    #     print(deltas)
    #     for name, param in self.algorithm.model.named_parameters():
    #         print(name)
    #         if name in deltas:
    #             # 检查 param 是否需要梯度
    #             if param.requires_grad:
    #                 term = torch.sum(param * deltas[name].detach())
    #                 surrogate_loss = term if surrogate_loss is None else surrogate_loss + term
    #             else:
    #                 # 输出警告，说明该参数未开启梯度追踪
    #                 print(f"Warning: {name} does not require grad.")

    #     # if surrogate_loss is None:
    #     #     raise ValueError("No matching model parameters found in deltas; surrogate loss is not computed.")

    #     self.hyper_optimizer.zero_grad()
    #     surrogate_loss.backward()
    #     self.hyper_optimizer.step()
    #     print("更新前的超网络输出",self.current_hsn_output)
    #     print("更新后的超网络输出：", self.hsn())

    #     self.current_hsn_output = self.hsn()
    #     import time
    #     time.sleep(60000)
    # def update_hsn(self,deltas,binary_masks): 
    #     deltas = deltas[0]
    #     for name, param in self.algorithm.model.named_parameters(): 
    #         if name in deltas: 
    #             param.grad = deltas[name] # 直接使用 deltas 作为梯度 # 构造虚拟损失以触发梯度计算（确保 soft_mask 参与计算图） 
    #     params = dict(self.algorithm.model.named_parameters())
    #     dummy_loss = 0.0 
    #     for key, value in binary_masks.items(): 
    #         if not isinstance(value, torch.Tensor): 
    #             value = torch.from_numpy(value) 
    #         value.requires_grad = True # 确保 value 需要梯度 # 确保 params[key+".weight"] 是 PyTorch 张量 
    #         if not isinstance(params[key+".weight"], torch.Tensor): 
    #             params[key+".weight"] = torch.from_numpy(params[key+".weight"]) 
    #         params[key+".weight"].requires_grad = True # 确保 params 需要梯度 # 
    #         print(value) 
    #         dummy_loss += torch.sum(params[key+".weight"] * value.view(-1, 1, 1, 1)) 
    #     self.hyper_optimizer.zero_grad() 
    #     dummy_loss.backward() # 触发反向传播 
    #     self.hyper_optimizer.step() 
    #     print(self.hsn())
    #     print(self.current_hsn_output)
    #     import time
    #     time.sleep(6000)
    from typing import OrderedDict, List
    # 然后直接使用 OrderedDict[str, torch.Tensor]

    def update_hsn(self,
                            diff: OrderedDict[str, torch.Tensor],
                            probab_masks,
                            save_graph = True,
                            retain_blocks: List[str] = []) -> None:
        import torch.autograd
        torch.autograd.set_detect_anomaly(True)

        print(probab_masks.keys())
        print(probab_masks["layer1.0.conv1"].grad_fn)
        #    full_state 为原始的客户端参数（通常为一个字典），binary_masks 为每层 mask
        # new_client_state = client_state_dict(self.algorithm.model, probab_masks)
        # model = self.model()
        # model.load_state_dict(new_client_state)
        # model.cuda()
        model = MaskedResNet(base_model=self.algorithm.model,probab_masks=probab_masks)
        model.cuda()
        criterion = torch.nn.CrossEntropyLoss()
        Predict_loss = 0.0
        

        # 假设整个 proxy_data_trainloader 的数据作为一个大的有效批次
        total_loss_for_backward = 0.0
        for batch_idx, (images, labels) in enumerate(self.proxy_data_trainloader):
            images, labels = images.to("cuda:0"), labels.to("cuda:0")
            log_probs = model(images)
            loss = criterion(log_probs, labels) * Config().parameters.loss_rate[0]
            # loss.backward(retain_graph=True)
            # total_loss_for_backward += loss
            # Predict_loss += loss.item()
            self.hyper_optimizer.zero_grad()
            if batch_idx != len(self.proxy_data_trainloader) - 1:
                loss.backward(retain_graph=True)
            else:
                loss.backward()
            self.hyper_optimizer.step()
            images, labels = images.cpu(), labels.cpu()
            del log_probs,loss
        # print(f"Predict_loss: {Predict_loss}")
        # total_loss_for_backward = Config().parameters.loss_rate[0] * total_loss_for_backward + Config().parameters.loss_rate[1] * self.reg_loss
        _,self.reg_loss = self.hsn()
        self.hyper_optimizer.zero_grad()
        total_loss_for_backward = Config().parameters.loss_rate[1] * self.reg_loss
        total_loss_for_backward.backward()
        self.hyper_optimizer.step()
        
        torch.cuda.empty_cache()  # 清理缓存


        self.current_hsn_output,self.reg_loss = self.hsn()
        print("超网络更新后：", self.current_hsn_output)
        self.save_hsn_outputs_to_txt(self.current_hsn_output, Config().parameters.hsn_outputs,self.current_round)
        

    def save_hsn_outputs_to_txt(self, hsn_outputs, file_path, current_round):
        # 将 hsn_outputs 转换为字符串表示，不改变原变量
        
        
        output_str = str(hsn_outputs)
        
        with open(file_path, "a") as f:
            f.write(f"=== Round {current_round} ===\n")
            f.write(output_str)
            f.write("\n")
        print(f"HSN outputs for round {current_round} saved to {file_path}.")
    def summon_clients(self,num_users):
        clients = []  # Every client is a tuple, miu ,sigma
        client_hetero_ration = list(map(float, Config().parameters.Parameters_to_percent_of_clients))
        users25 = int(num_users * round(client_hetero_ration[0] / sum(client_hetero_ration), 2))
        users50 = int(num_users * round(client_hetero_ration[1] / sum(client_hetero_ration), 2))
        users100 = int(num_users * round(client_hetero_ration[2] / sum(client_hetero_ration), 2))

        # if args.r == 0:
        #     for i in range(users25):
        #         clients.append((35, random.choice([5, 8, 10])))
        #     for i in range(users50):
        #         clients.append((60, random.choice([5, 8, 10])))
        #     for i in range(users100):
        #         clients.append((110, random.choice([5, 8, 10])))
        #     return clients
        # elif args.r == 1:
        #     for i in range(users25):
        #         clients.append((35, random.choice([0])))
        #     for i in range(users50):
        #         clients.append((60, random.choice([0])))
        #     for i in range(users100):
        #         clients.append((110, random.choice([0])))
        #     return clients
        # elif args.r == 2:
        for i in range(users25):
            clients.append((35, random.choice([10, 20, 30])))
        for i in range(users50):
            clients.append((60, random.choice([10, 20, 30])))
        for i in range(users100):
            clients.append((110, random.choice([10, 20, 30])))
        return clients
    def resource_to_model3(self,resource):
        if resource < 50:
            model = 0.25
        elif resource < 100:
            model = 0.5
        else:
            model = 1
        return model







        

    