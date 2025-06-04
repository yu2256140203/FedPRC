from client import client
from server import server
from algorithm import FedPRC_algorithm
from trainer import FedPRC_trainer
from plato.config import Config
import resnet
import mobileNetV2
import vgg
from plato.config import Config

def main():

    if Config().parameters.model == "resnet18":
        if Config.data.datasource == "CIFAR10":
            model = resnet.resnet18
        elif Config.data.datasource == "CIFAR100":
            model = resnet.resnet18_CIFAR100
        elif Config.data.datasource == "TinyImagenet":
            model = resnet.resnet18_TinyImagenet
    elif Config().parameters.model == "resnet34":
        if Config.data.datasource == "CIFAR10":
            model = resnet.resnet34
        elif Config.data.datasource == "CIFAR100":
            model = resnet.resnet34_CIFAR100
        elif Config.data.datasource == "TinyImagenet":
            model = resnet.resnet34_TinyImagenet
    # elif Config().parameters.model == "mobileNetV2":
    #     if Config.data.datasource == "CIFAR10":
    #         model = mobileNetV2.resnet34
    #     elif Config.data.datasource == "CIFAR100":
    #         model = mobileNetV2.resnet34_CIFAR100
    #     elif Config.data.datasource == "TinyImagenet":
    #         model = mobileNetV2.resnet34_TinyImagenet
    elif Config().parameters.model == "vgg":
        if Config.data.datasource == "CIFAR10":
            model = vgg.vgg_16_bn
        elif Config.data.datasource == "CIFAR100":
            model = vgg.vgg_16_bn_CIFAR100
        elif Config.data.datasource == "TinyImagenet":
            model = vgg.vgg_16_bn_TinyImagenet


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