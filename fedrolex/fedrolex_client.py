"""
Customized Client for FedRolex.
"""
from plato.config import Config
from plato.clients import simple

import logging
import os
class Client(simple.Client):
    """A federated learning server using the FedRolexFL algorithm."""

    def process_server_response(self, server_response) -> None:
        rate = server_response["rate"]
        self.algorithm.model = self.model(
            model_rate=rate, **Config().parameters.client_model._asdict()
        )
        print("当前客户端:",self.client_id,"参数量：",self.count_parameters(self.algorithm.model))
        self.trainer.model = self.algorithm.model
    def count_parameters(self,model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    async def _start_training(self, inbound_payload):
        """Complete one round of training on this client."""
        self._load_payload(inbound_payload)

        report, outbound_payload = await self._train()
        # outbound_payload = {"outbound_payload":outbound_payload,"client_id":self.client_id}

        if Config().is_edge_server():
            logging.info(
                "[Server #%d] Model aggregated on edge server (%s).", os.getpid(), self
            )
        else:
            logging.info("[%s] Model trained.", self)

        return report, outbound_payload
