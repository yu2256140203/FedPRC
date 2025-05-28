from client import client
from server import server
from algorithm import FedPRC_algorithm
from trainer import FedPRC_trainer
from plato.config import Config
import resnet
from plato.config import Config

def main():


    # 创建模型实例
    model = resnet.resnet18
    if Config().parameters.model == "resnet34":
        model = resnet.resnet34


    # 创建算法实例
    algorithm = FedPRC_algorithm

    # 创建训练器实例
    trainer = FedPRC_trainer

    # 创建服务器和客户端实例
    Server = server(trainer=FedPRC_trainer, model=model, algorithm=algorithm)
    Client = client(trainer=FedPRC_trainer, model=model, algorithm=algorithm)

    # 启动服务器
    Server.run(Client)

if __name__ == "__main__":
    main()