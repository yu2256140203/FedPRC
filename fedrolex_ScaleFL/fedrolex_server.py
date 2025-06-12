"""
FedRolexFL algorithm trainer.
"""

import numpy as np

from plato.config import Config
from plato.servers import fedavg


class Server(fedavg.Server):
    """A federated learning server using the FedRolexFL algorithm."""

    def __init__(
        self,
        model=None,
        datasource=None,
        algorithm=None,
        trainer=None,
    ):
        # pylint:disable=too-many-arguments
        super().__init__(model, datasource, algorithm, trainer)
        self.rates = [None for _ in range(Config().clients.total_clients)]
        self.limitation = np.zeros(
            (Config().trainer.rounds, Config().clients.total_clients, 2)
        )
        if (
            hasattr(Config().parameters.limitation, "activated")
            and Config().parameters.limitation.activated
        ):
            limitation = Config().parameters.limitation
            self.limitation[:, :, 0] = np.random.uniform(
                limitation.min_size,
                limitation.max_size,
                (Config().trainer.rounds, Config().clients.total_clients),
            )
            self.limitation[:, :, 1] = np.random.uniform(
                limitation.min_flops,
                limitation.max_flops,
                (Config().trainer.rounds, Config().clients.total_clients),
            )
                # 获取配置中的参数
        rates_values = Config().parameters.rates
        parameters_to_clients_ratio = Config().parameters.Parameters_to_percent_of_clients
        self.prune_rates = [None for _ in range(Config().clients.total_clients)]
        
        # 根据 Parameters_to_percent_of_clients 的比例分配客户端
        total_clients = Config().clients.total_clients
        clients_per_group = total_clients // len(parameters_to_clients_ratio)
        self.deltas = None
        
        for client_idx in range(total_clients):
            if client_idx < (total_clients * parameters_to_clients_ratio[0]) // sum(parameters_to_clients_ratio):
                group_index = 0
            elif client_idx < total_clients * (parameters_to_clients_ratio[0]+ parameters_to_clients_ratio[1]) // sum(parameters_to_clients_ratio):
                group_index = 1
            else:
                group_index = 2
            # 设置 rates
            self.rates[client_idx] = rates_values[group_index]

    def customize_server_response(self, server_response: dict, client_id) -> dict:
        # rate = self.algorithm.choose_rate(
        #     self.limitation[self.current_round - 1, client_id - 1], self.model
        # )
        server_response["rate"] =  self.rates[client_id - 1]
        rate = server_response["rate"]
        #已经修改了
        rate = self.algorithm.choose_rate_new(
            self.model,rate
        )
        return super().customize_server_response(server_response, client_id)

    async def aggregate_weights(
        self, updates, baseline_weights, weights_received
    ):  # pylint: disable=unused-argument
        """Aggregates weights of models with different architectures."""

        return self.algorithm.aggregation(weights_received)

    def weights_aggregated(self, updates):
        super().weights_aggregated(updates)
        self.algorithm.sort_channels()
