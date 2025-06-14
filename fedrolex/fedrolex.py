"""
A federated learning training session using FedRolexFL

Alam, Samiul and Liu, Luyang and Yan, Ming and Zhang, Mi
“FedRolex: Model-Heterogeneous Federated Learning with Rolling Sub-Model Extraction,”
in FedRolex NIPS2022.

"""

from resnet import resnet18,resnet34
from vit import ViT

from fedrolex_client import Client
from fedrolex_server import Server
from fedrolex_algorithm import Algorithm
from fedrolex_trainer import ServerTrainer
from plato.config import Config


def main():
    # import time
    # time.sleep(14400)
    """A Plato federated learning training session using the FedRolexFL algorithm."""
    if "resnet18" in Config().trainer.model_name:
        model = resnet18
    elif "resnet34" in Config().trainer.model_name:
        model = resnet34
    else:
        model = ViT
    server = Server(model=model, algorithm=Algorithm, trainer=ServerTrainer)
    client = Client(model=model)
    server.run(client)


if __name__ == "__main__":
    main()
