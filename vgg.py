"""vgg in pytorch
[1] Karen Simonyan, Andrew Zisserman
    Very Deep Convolutional Networks for Large-Scale Image Recognition.
    https://arxiv.org/abs/1409.1556v6
"""
import math

import numpy as np
import torch
import resnet

'''VGG11/13/16/19 in Pytorch.'''

import torch.nn as nn

cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512]
}


class VGG(nn.Module):
    def __init__(self, features, num_class=100, num_channels=3, rate=None):
        super().__init__()
        self.features = features

        if num_channels == 3:
            dim = 4096
        else:
            dim = 256
        if num_class == 200:
            self.features[-1].append(nn.AdaptiveAvgPool2d((1, 1)))
        self.features[-1].append(nn.Flatten(start_dim=1, end_dim=-1))
        self.features.append(nn.Sequential(
            nn.Linear(int(512 * rate[-3]), int(dim * rate[-2])),
            nn.ReLU(inplace=True),
            nn.Dropout()))
        self.features.append(nn.Sequential(
            nn.Linear(int(dim * rate[-2]), int(dim * rate[-1])),
            nn.ReLU(inplace=True),
            nn.Dropout()))
        self.classifier = nn.Linear(int(dim * rate[-1]), num_class)
        # self.reset_parameters()

    def forward(self, x):

        output = self.features(x)
        # result = {'representation': output}
        output = self.classifier(output)
        # result['output'] = output
        return output
    
    def __call__(self, x=None):
        if x == None:
            return self
        else:
            return self.forward(x)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()


def make_layers(cfg, batch_norm=False, track_running_stats=True, num_channels=3, rate=None):
    layers = []
    input_channel = num_channels
    maxpool = None
    if num_channels == 3:
        cfg.append("M")
    index = 0
    for l in cfg:
        if l == 'M':
            maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
            continue
        if index == 0:
            conv2d = nn.Conv2d(input_channel, int(l * rate[index]), kernel_size=3, padding=1)
        else:
            conv2d = nn.Conv2d(int(input_channel * rate[index - 1]), int(l * rate[index]), kernel_size=3, padding=1)
        if batch_norm:
            seq = nn.Sequential(
                conv2d,
                nn.BatchNorm2d(int(l * rate[index]), track_running_stats=track_running_stats),
                nn.ReLU(inplace=True)
            )
        else:
            seq = nn.Sequential(
                conv2d,
                nn.ReLU(inplace=True)
            )

        if maxpool is not None:
            seq.add_module('MaxPool2d', maxpool)
            maxpool = None

        layers.append(seq)
        input_channel = l
        index += 1

    # 如果最后一个元素是 'M'，那么 maxpool 不会是 None，我们需要将其添加到 layers 中
    if maxpool is not None:
        layers[-1].append(maxpool)

    return nn.Sequential(*layers)


def vgg_16_bn(classes_size=10, track_running_stats=True, num_channels=3, layer_prune_rates=None):
    if layer_prune_rates is not None:
        layer_prune_rates.insert(0, 1)  #保证第一层卷积层不剪枝
    if layer_prune_rates is None:
        layer_prune_rates = [1.0] * len(cfg['D'])
    return VGG(
        make_layers(cfg['D'], batch_norm=True, track_running_stats=track_running_stats, num_channels=num_channels,
                    rate=layer_prune_rates),
        num_class=classes_size,
        num_channels=num_channels, rate=layer_prune_rates)
def vgg_16_bn_CIFAR100(classes_size=100, track_running_stats=True, num_channels=3, layer_prune_rates=None):
    if layer_prune_rates is not None:
        layer_prune_rates.insert(0, 1)  #保证第一层卷积层不剪枝
    if layer_prune_rates is None:
        layer_prune_rates = [1.0] * len(cfg['D'])
    return VGG(
        make_layers(cfg['D'], batch_norm=True, track_running_stats=track_running_stats, num_channels=num_channels,
                    rate=layer_prune_rates),
        num_class=classes_size,
        num_channels=num_channels, rate=layer_prune_rates)
def vgg_16_bn_TinyImagenet(classes_size=200, track_running_stats=True, num_channels=3, layer_prune_rates=None):
    if layer_prune_rates is not None:
        layer_prune_rates.insert(0, 1)  #保证第一层卷积层不剪枝
    if layer_prune_rates is None:
        layer_prune_rates = [1.0] * len(cfg['D'])
    return VGG(
        make_layers(cfg['D'], batch_norm=True, track_running_stats=track_running_stats, num_channels=num_channels,
                    rate=layer_prune_rates),
        num_class=classes_size,
        num_channels=num_channels, rate=layer_prune_rates)

if __name__ == '__main__':
    # model = vgg_16_bn_CIFAR100(100, True, 3,
    #                     [1] * 50)
    model = vgg_16_bn_CIFAR100()
    print(model)
    # # Random summon data to test model_1(CIFAR10)
    # data = torch.rand(10, *(3, 64, 64))
    # totalParam = []

    # model = vgg_16_bn(10, True, 3,
    #                   [0.71]*15)
    # LayerParams = np.array([])
    # for layer in model_1.features:
    #     params = sum(p.numel() for p in layer.parameters())
    #     totalParam = np.append(totalParam, params)
    # for layer in model.features:
    #     params = sum(p.numel() for p in layer.parameters())
    #     LayerParams = np.append(LayerParams, params)
    # totalParam = np.append(totalParam, sum(p.numel() for p in model_1.classifier.parameters()))
    # LayerParams = np.append(LayerParams, sum(p.numel() for p in model.classifier.parameters()))

    # print(list(LayerParams/totalParam))
    from torchvision.datasets import CIFAR100
        # 加载 CIFAR10 数据集（训练集和测试集均适用，不变）
    from torchvision.transforms import Compose, ToTensor, Normalize
    from torch.utils.data import DataLoader
    from plato.config import Config
    import random
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_dataset = CIFAR100(root="./data", train=True, download=True, transform=transform)
    test_dataset  = CIFAR100(root="./data", train=False, download=True, transform=transform)
    train_loader  = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader   = DataLoader(test_dataset, batch_size=32, shuffle=False)
    criterion = torch.nn.CrossEntropyLoss()
    # model.cuda()
    total_loss_for_backward = 0.0
    for batch_idx, (images, labels) in enumerate(train_loader):
        # images, labels = images.to("cuda:0"), labels.to("cuda:0")
        print(images.shape)
        log_probs = model(images)
        print(log_probs.shape)
        print(labels.shape)
        loss = criterion(log_probs, labels)
        # images, labels = images.cpu(), labels.cpu()
        del log_probs,loss