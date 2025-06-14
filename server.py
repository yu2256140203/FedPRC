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
from prune import restore_client_full_state,get_channel_indices_unuse_hsn,client_state_dict,prune_state_dict,reverse_prune_rates,get_channel_indices,aggregate_submodel_states,prune_mapping_with_global_threshold_and_binary_indices,get_model_param_output_channels
from channel_evaluation import HyperStructureNetwork,combine_importance,MaskedResNet,ChannelImportanceEvaluator
from torch.utils.data import Subset
import wandb
from utils.statistics import transfer_optimizer_to_new_model



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

        #名字与hsn_outputs相同
        wandb.init(project="FedPRC_ResNet34_CIFAR10",name=Config.parameters.hsn_outputs.split("/", 1)[1].rsplit(".", 1)[0].replace("/", "_"), config=Config())

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model
        self.server_importance_dicts = {name: [0.0] * module.out_channels for name, module in model().named_modules() 
        if 'conv' in name} #初始化每层的通道重要性
        #我并不需要在全局直接算出来当前的full_model_global_ranks，毕竟按照通道重要性合并完，再计算也一样
        if Config().data.datasource == "CIFAR10" or Config().data.datasource == "CIFAR100":
            self.input_sizes = [32,32]
        else:
            self.input_sizes = [64,64]
        
        self.channel_importance_dicts = [None for _ in range(Config().clients.total_clients)]#本地当前轮次全局重要性排名
        self.client_losses = []
        self.client_full_model_global_ranks = [None for _ in range(Config().clients.total_clients)]
        #客户端的每个通道对于完整模型的重要性排名
        self.clients_mapping_indices = [None for _ in range(Config().clients.total_clients)]
        self.probab_masks = None

        # self.current_linked_client_ids = None # 当前轮次的客户端id，但是有序的，按照通信顺序来的
        self.rates = [0 for _ in range(Config().clients.total_clients)]
        self.modules_indices = get_model_param_output_channels(model)

        self.init_mapping = get_channel_indices(modules_indices=self.modules_indices)
        
        # 获取配置中的参数
        self.rates_values = Config().parameters.rates
        parameters_to_clients_ratio = Config().parameters.Parameters_to_percent_of_clients
        self.prune_rates = [None for _ in range(Config().clients.total_clients)]
        
        # 根据 Parameters_to_percent_of_clients 的比例分配客户端
        total_clients = Config().clients.total_clients
        clients_per_group = total_clients // len(parameters_to_clients_ratio)
        self.deltas = None
        
        #随机的三种mapping
        self.current_mapping = {}
        self.current_prune_rates = {}
        self.client_prune_rates = [None for _ in range(Config().clients.total_clients)]

        for client_idx in range(total_clients):
            if client_idx < (total_clients * parameters_to_clients_ratio[0]) // sum(parameters_to_clients_ratio):
                group_index = 0
            elif client_idx < total_clients * (parameters_to_clients_ratio[0]+ parameters_to_clients_ratio[1]) // sum(parameters_to_clients_ratio):
                group_index = 1
            else:
                group_index = 2
            # 设置 rates
            self.rates[client_idx] = self.rates_values[group_index]


        if Config().parameters.FlexFL_exp:
            self.rates = [0 for _ in range(Config().clients.total_clients)]
            self.clients_FLexFL_exp  = self.summon_clients(total_clients)
            


        self.train_model = None
        # self.conv_names = [i for i in model().state_dict().keys() if 'conv' in i and i != Config().parameters.first_layer]

        conv_names = [
            name for name, module in model().named_modules()
            if (isinstance(module, torch.nn.Conv2d) or isinstance(module,torch.nn.Linear)) and "shortcut" not in name
        ]
        # if Config().parameters.model == "resnet18" or Config().parameters.model == "resnet34":
        self.conv_names = conv_names[1:-1]        # elif Config().parameters.model == "vgg":
        #     self.conv_names = conv_names[1:-3]
        


        
        num_layers = len(self.conv_names)
        # self.hsn = HyperStructureNetwork(num_layers=num_layers, d=16, gru_hidden_dim=16, h_dim=32, p=0.5, tau=1.0).cuda()
        self.hsn = HyperStructureNetwork(num_layers=num_layers, d=16, gru_hidden_dim=16, h_dim=32, p=0.5).cuda()

        self.current_hsn_output,self.reg_loss = self.hsn()
        # self.hyper_optimizer  = torch.optim.SGD(self.hsn.parameters(), lr=0.01,momentum=0.9)
        self.hyper_optimizer  = torch.optim.SGD(self.hsn.parameters(), lr=0.1)
        self.dict_modules = {}
        for name, module in model().named_modules():
            if isinstance(module, torch.nn.Conv2d):
                self.dict_modules[name] = "conv"
            elif isinstance(module,torch.nn.BatchNorm2d):
                self.dict_modules[name] = "bn"
            elif isinstance(module,torch.nn.Linear):
                self.dict_modules[name] = "linear"
        
        for i in range(30):
                self.hyper_optimizer.zero_grad()
                self.reg_loss.backward(retain_graph=True)  # 触发反向传播
                self.hyper_optimizer.step()
                #初始的超网络输出：
                self.current_hsn_output,self.reg_loss = self.hsn()
                print(self.current_hsn_output)
        if Config().parameters.momentum == 0:
            self.hyper_optimizer  = torch.optim.SGD(self.hsn.parameters(), lr=0.1)
        else:
            self.hyper_optimizer  = torch.optim.SGD(self.hsn.parameters(), lr=0.1,momentum=Config().parameters.momentum)
        # self.hyper_optimizer  = torch.optim.SGD(self.hsn.parameters(), 
        #                     lr=0.1,
        #                     momentum=0.5, 
        #                     weight_decay=0.0001)
        # self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.hyper_optimizer , gamma=0.9)

        #重新定义超网络优化器，探索重要性
        #希望初始超网络的输出能公平少一点随机，这样让初始的模型能有较好的性能
        self.acc = [0.0 for _ in range(Config().clients.total_clients)]#本地当前轮次全局重要性排名
        self.proxy_data_trainloader = None
        self.proxy_data_testloader = None
        self.client_data_sizes = [0 for _ in range(Config().clients.total_clients)]
        
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
            evaluator = ChannelImportanceEvaluator(self.algorithm.model, self.device)
            server_importance_dicts = evaluator.evaluate(self.proxy_data_trainloader, threshold=0.5)
            for name in self.conv_names:#保证只剪枝规定的层 VGG的多个线性层问题
                self.server_importance_dicts[name] = server_importance_dicts[name]

            self.clients_mapping_indices[self.selected_client_id-1] = get_channel_indices_unuse_hsn(model=self.model(),submodel_layer_prune_rate=rate,importance=self.server_importance_dicts,dict_modules=self.dict_modules,modules_indices=self.modules_indices)
            server_response["mapping_indices"] = self.clients_mapping_indices[self.selected_client_id-1]
            # self.clients_mapping_indices[self.selected_client_id-1] = server_response["mapping_indices"]
        elif Config().parameters.unuse_hsn != None and Config().parameters.unuse_hsn == True:
            self.clients_mapping_indices[self.selected_client_id-1] = get_channel_indices_unuse_hsn(model=self.model(),submodel_layer_prune_rate=rate,importance=self.server_importance_dicts,dict_modules=self.dict_modules,modules_indices=self.modules_indices)
            server_response["mapping_indices"] = self.clients_mapping_indices[self.selected_client_id-1]
            
        else:
            self.client_full_model_global_rank = combine_importance(importance_dict=self.server_importance_dicts, hsn_outputs=self.current_hsn_output, device=self.device, conv_names=self.conv_names)
            
            pruned_mapping,binary_masks, probab_masks, binary_indices, channels_info_sorted, total_param, target_keep  =prune_mapping_with_global_threshold_and_binary_indices(initial_mapping=self.init_mapping,
            final_importance=self.client_full_model_global_rank, model=self.model(),target_ratio=rate * rate, device=self.device,dict_modules=self.dict_modules,modules_indices=self.modules_indices)
            self.clients_mapping_indices[self.selected_client_id-1 ] = pruned_mapping
            self.probab_masks = probab_masks
            server_response["mapping_indices"] = self.clients_mapping_indices[self.selected_client_id-1 ]
        server_response["prune_rates"] = reverse_prune_rates(server_response["mapping_indices"], dict_modules=self.dict_modules,modules_indices = self.modules_indices)
        self.client_prune_rates[self.selected_client_id-1 ] = server_response["prune_rates"]
        if str(rate) not in self.current_mapping.keys():
            self.current_mapping[str(rate)] = server_response["mapping_indices"]
            self.current_prune_rates[str(rate)] = server_response["prune_rates"]


        

        return super().customize_server_response(server_response, client_id)
    
    async def aggregate_weights(
        self, updates, baseline_weights, weights_received
    ):  # pylint: disable=unused-argument
        # 更新当前获得的重要性
        for payload in weights_received:
            self.acc[payload["client_id"] - 1] = payload["acc"]
            self.client_data_sizes[payload["client_id"]-1] = payload["data_size"]
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

                
                # # 更新 server_importance_dicts 的值为计算后的 avg_tensor
                # self.server_importance_dicts[key] = avg_tensor
                                # 更新 server_importance_dicts 的值为计算后的 avg_tensor
                # 将原来的 server_importance_dicts[key] 转换为 tensor
                old_value_tensor = torch.tensor(self.server_importance_dicts[key], device=self.device)
                avg = 0.5 * (0.9 ** (self.current_round / Config().parameters.ranks_round-1))
                old = 1-avg
                # 保证两边都是 Tensor 后进行计算：一半用计算得到的 avg_tensor，一半保留原来的值
                self.server_importance_dicts[key] = avg * avg_tensor + old * old_value_tensor

                
                # mean_imp = self.server_importance_dicts[key].mean()
                # std_imp = self.server_importance_dicts[key].std() + 1e-6
                # self.server_importance_dicts[key] = (self.server_importance_dicts[key] - mean_imp) / std_imp  # Tensor, (C,)

                # self.server_importance_dicts[key] = self.server_importance_dicts[key].detach().cpu().numpy()
               
            print("当前完成重要性聚合")
        print("当前完成聚合")
        
        last_global_parameters = copy.deepcopy(self.algorithm.extract_weights())
        """Aggregates weights of models with different architectures."""

        # KD_client_weights = []

        # for index,payload in enumerate(weights_received):
        #     student_model = self.model()
        #     student_model_state_dict = restore_client_full_state(full_state=last_global_parameters,sub_state_list=[payload["outbound_payload"]],mapping_indices_list=[payload["mapping_indices"]])
        #     student_model.load_state_dict(student_model_state_dict)
        #     teacher_model = self.model(self.client_prune_rates[payload["client_id"]-1])
        #     teacher_model.load_state_dict(payload["outbound_payload"])
        #     student_model = self.Model_KD(model=student_model,teacher_model=teacher_model,proxy_data_trainloader=self.proxy_data_trainloader)
        #     KD_client_weights.append((copy.deepcopy(student_model.state_dict()),payload["data_size"]))
        # global_parameters = self.algorithm.aggregation(KD_client_weights)
        global_parameters = self.algorithm.aggregation(weights_received)

        #获取更新超网络的deltas
        #yu 当前所有的客户端的mask应当是一致的，所以可以只调用一次超网络反向传播
        # if  Config().parameters.unuse_hsn != None and  Config().parameters.unuse_hsn != True and  self.current_round % Config().parameters.ranks_round == 0 and self.current_round != Config().parameters.ranks_round:
        if  Config().parameters.unuse_hsn != None and  Config().parameters.unuse_hsn != True and  self.current_round % Config().parameters.ranks_round == 0 and self.current_round != 1:

            self.deltas  = None
            # for idx, (client_parameters, payload) in enumerate(zip(restored_models, weights_received)):
            for idx, payload in enumerate(weights_received):

                # 强行转化成列表
                # self.deltas = copy.deepcopy(self.algorithm.compute_weight_deltas(last_global_parameters, [client_parameters]))
                
                # if self.probab_masks[payload["client_id"] - 1] is not None and idx == len(weights_received) - 1:
                #     #如果是最后一项，释放掉计算图
                # self.update_hsn(self.deltas[0], self.probab_masks[payload["client_id"] - 1])
                self.update_hsn(self.probab_masks)
                break

                    
        self.trainer.model.load_state_dict(global_parameters)
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
            # print(self.avg_acc,self.large_acc,self.middle_acc,self.small_acc)
            if len(self.avg_acc) == 0:
                avg_acc = 0.0
            else:
                # print(sum(self.avg_acc), len(self.avg_acc))
                avg_acc = sum(self.avg_acc) / len(self.avg_acc)
            large_acc = self.acc_sub["1"]
            middle_acc = self.acc_sub["0.71"]
            small_acc  = self.acc_sub["0.5"]


            logged_items["avg_accuracy"] = avg_acc
            logged_items["large_accuracy"] = large_acc
            logged_items["middle_accuracy"] = middle_acc
            logged_items["small_accuracy"] = small_acc


            return logged_items
    
    def customize_server_payload(self, payload):
        #定制子模型
        payload = self.algorithm.sub_weights(payload,self.clients_mapping_indices[self.selected_client_id-1])
        # model = self.model(layer_prune_rates=self.client_prune_rates[self.selected_client_id-1])
        # model.load_state_dict(payload)
        # model = self.Model_KD(model,self.trainer.model,self.proxy_data_trainloader)
        # return model.state_dict()
        return payload

        # return super().customize_server_payload(payload)
    def Model_KD(self, model,teacher_model, proxy_data_trainloader, temperature=1.0, alpha=0.5, criterion=None):
        """
        Perform knowledge distillation using KL Divergence.
        model: the student model
        dataset: the dataset to be used for training
        temperature: scaling factor for softening the probabilities (default 1.0)
        alpha: weight factor for combining the hard and soft loss (default 0.5)
        criterion: the loss function (default is cross entropy)
        """
        import torch.nn.functional as F
        import time
        start_time = time.time()
        if criterion is None:
            criterion = torch.nn.KLDivLoss()
        teacher_model.cuda()
        model.cuda()
        model.train()  # set model to training mode
        optimizer = self.trainer.get_optimizer(model)
        lr_scheduler = self.trainer.get_lr_scheduler(Config().trainer, optimizer)
        optimizer = self.trainer._adjust_lr(Config().trainer,lr_scheduler,optimizer)
        total_loss = 0.0
        correct = 0
        total = 0

        for inputs, targets in proxy_data_trainloader:
            inputs, targets = inputs.cuda(), targets.cuda()

            # Get teacher model's output (soft labels)
            teacher_outputs = teacher_model(inputs)
            teacher_probs = F.softmax(teacher_outputs / temperature, dim=1)  # Softened outputs

            # Get student model's output (hard labels and features)
            student_outputs = model(inputs)
            student_probs = F.softmax(student_outputs / temperature, dim=1)  # Softened outputs

            # Calculate the KD loss (KL Divergence)
            soft_loss = F.kl_div(F.log_softmax(student_outputs / temperature, dim=1),
                                teacher_probs, reduction='batchmean') * (temperature ** 2)


            # Backpropagation
            optimizer.zero_grad()
            soft_loss.backward()
            optimizer.step()  # Assuming model has an optimizer
        end_time = time.time()


        print(f"蒸馏完成，用时",end_time-start_time )

        return model
    

    from typing import OrderedDict, List
    # 然后直接使用 OrderedDict[str, torch.Tensor]

    def update_hsn(self,
                            # diff: OrderedDict[str, torch.Tensor],
                            probab_masks,
                            save_graph = True,
                            retain_blocks: List[str] = []) -> None:
        import torch.autograd
        torch.autograd.set_detect_anomaly(True)

        # print(probab_masks.keys())
        # print(probab_masks["layer1.0.conv1"].grad_fn)
        #    full_state 为原始的客户端参数（通常为一个字典），binary_masks 为每层 mask
        # new_client_state = client_state_dict(self.algorithm.model, probab_masks)
        model = MaskedResNet(base_model=self.algorithm.model,hsn=self.hsn,important=self.server_importance_dicts,probab_masks=probab_masks)
        model.cuda()
        criterion = torch.nn.CrossEntropyLoss()
        # self.hyper_optimizer  = torch.optim.SGD(self.hsn.parameters(), lr=0.1*(0.9 ** (self.current_round/Config().parameters.ranks_round)))
        Predict_loss = 0.0
        

        # 假设整个 proxy_data_trainloader 的数据作为一个大的有效批次
        total_loss_for_backward = 0.0
        for batch_idx, (images, labels) in enumerate(self.proxy_data_trainloader):
            images, labels = images.to("cuda:0"), labels.to("cuda:0")
            # print(images.shape)
            log_probs = model(images)
            # print(log_probs.shape)
            # print(labels.shape)
            self.current_hsn_output,self.reg_loss = self.hsn()
            # loss = criterion(log_probs, labels) * Config().parameters.loss_rate[0]+Config().parameters.loss_rate[1] * self.reg_loss
            loss = criterion(log_probs, labels) * Config().parameters.loss_rate[0]
            self.hyper_optimizer.zero_grad()
            if batch_idx != len(self.proxy_data_trainloader) - 1:
                # loss.backward(retain_graph=True)
                loss.backward()

            else:
                loss.backward()
            self.hyper_optimizer.step()
            images, labels = images.cpu(), labels.cpu()
            del log_probs,loss
        
        _,self.reg_loss = self.hsn()
        self.hyper_optimizer.zero_grad()
        total_loss_for_backward = Config().parameters.loss_rate[1] * self.reg_loss
        total_loss_for_backward.backward()
        self.hyper_optimizer.step()

        # self.scheduler.step()
        
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
    async def _process_reports(self):#找不到为什么服务端测试准确率一直不好使
        """Process the client reports by aggregating their weights."""
        weights_received = [update.payload for update in self.updates]

        weights_received = self.weights_received(weights_received)
        self.callback_handler.call_event("on_weights_received", self, weights_received)

        # Extract the current model weights as the baseline
        baseline_weights = self.algorithm.extract_weights()

        if hasattr(self, "aggregate_weights"):
            # Runs a server aggregation algorithm using weights rather than deltas
            logging.info(
                "[Server #%d] Aggregating model weights directly rather than weight deltas.",
                os.getpid(),
            )
            updated_weights = await self.aggregate_weights(
                self.updates, baseline_weights, weights_received
            )

            # Loads the new model weights
            self.algorithm.load_weights(updated_weights)
        else:
            # Computes the weight deltas by comparing the weights received with
            # the current global model weights
            deltas_received = self.algorithm.compute_weight_deltas(
                baseline_weights, weights_received
            )
            # Runs a framework-agnostic server aggregation algorithm, such as
            # the federated averaging algorithm
            logging.info("[Server #%d] Aggregating model weight deltas.", os.getpid())
            deltas = await self.aggregate_deltas(self.updates, deltas_received)
            # Updates the existing model weights from the provided deltas
            updated_weights = self.algorithm.update_weights(deltas)
            # Loads the new model weights
            self.algorithm.load_weights(updated_weights)

        # The model weights have already been aggregated, now calls the
        # corresponding hook and callback
        self.weights_aggregated(self.updates)
        self.callback_handler.call_event("on_weights_aggregated", self, self.updates)

        # Testing the global model accuracy
        if hasattr(Config().server, "do_test") and not Config().server.do_test:
            # Compute the average accuracy from client reports
            self.accuracy, self.accuracy_std = self.get_accuracy_mean_std(self.updates)
            logging.info(
                "[%s] Average client accuracy: %.2f%%.", self, 100 * self.accuracy
            )
        else:
            # Testing the updated model directly at the server
            logging.info("[%s] Started model testing.", self)
            rates = self.current_mapping.keys()
            self.acc_sub = {}
            parameters_list = self.algorithm.get_test_parameters(self.trainer.model,self.current_mapping)
            for index,i in enumerate(rates):
                model = self.model(layer_prune_rates=self.current_prune_rates[i])
                model.load_state_dict(parameters_list[i])
                self.acc_sub[i] = self.trainer.custom_server_test(self.testset,model)
            real_rates = Config().parameters.rates
            for rate in  real_rates:
                if str(rate) not in rates:
                    pruned_mapping,binary_masks, probab_masks, binary_indices, channels_info_sorted, total_param, target_keep  =prune_mapping_with_global_threshold_and_binary_indices(initial_mapping=self.init_mapping,
            final_importance=self.client_full_model_global_rank, model=self.model(),target_ratio=rate * rate, device=self.device,dict_modules=self.dict_modules,modules_indices=self.modules_indices)
                    current_prune_rate = reverse_prune_rates(pruned_mapping, dict_modules=self.dict_modules,modules_indices = self.modules_indices)
                    model = self.model(layer_prune_rates=current_prune_rate)
                    parameters = self.algorithm.get_local_parameters(self.trainer.model.state_dict(),pruned_mapping)
                    model.load_state_dict(parameters)
                    self.acc_sub[str(rate)] = self.trainer.custom_server_test(self.testset,model)
            self.accuracy = self.trainer.custom_server_test(self.testset,self.trainer.model)
            self.current_mapping = {}#清空等待下一轮
            self.current_prune_rates = {}
            # 假设在训练变量中 round 对应当前训练轮次
            logged_items = self.get_logged_items() 
            wandb.log(logged_items)

            # self.accuracy = self.trainer.custom_server_test(self.testset)

        if hasattr(Config().trainer, "target_perplexity"):
            logging.info(
                fonts.colourize(
                    f"[{self}] Global model perplexity: {self.accuracy:.2f}\n"
                )
            )
        else:
            logging.info(
                fonts.colourize(
                    f"[{self}] Global model accuracy: {100 * self.accuracy:.2f}%\n"
                )
            )

        self.clients_processed()
        self.callback_handler.call_event("on_clients_processed", self)







        

    