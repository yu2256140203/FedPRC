"""
修改后的ResNet模型代码，解决了剪枝率不同导致的结构不一致问题。
主要修改：
1. 统一模型结构，始终创建shortcut连接
2. 使用条件身份映射，在不需要shortcut时直接使用输入
"""

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from plato.config import Config

def init_param(model):
    "Initialize the parameters of resnet."
    if isinstance(model, (nn.BatchNorm2d, nn.InstanceNorm2d)):
        model.weight.data.fill_(1)
        model.bias.data.zero_()
    elif isinstance(model, nn.Linear):
        model.bias.data.zero_()
    return model

class Scaler(nn.Module):
    "A simple scaling layer. In training, scales input by dividing with rate; in inference, unchanged."
    def __init__(self, rate):
        super().__init__()
        self.rate = rate

    def forward(self, feature):
        return feature / self.rate if self.training else feature

# -----------------------------
# Modified Basic Block (Block)
# -----------------------------
class Block(nn.Module):
    """
    Modified ResNet basic block supporting per-layer prune rates.
    每个 Block 包含两个卷积层，通过传入剪枝率 [r1, r2]，输出通道数分别为：
      conv1: new_conv1 = ceil(planes * r1)
      conv2: new_conv2 = ceil(planes * r2)
    最终 Block 输出通道数为 new_conv2.
    
    修改点：始终创建shortcut连接，但在forward中根据need_shortcut决定是否使用。
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride, prune_rates, track):
        super().__init__()
        new_conv1 = int(np.ceil(planes * prune_rates[0]))
        new_conv2 = int(np.ceil(planes * prune_rates[1]))
        self.out_channels = new_conv2

        self.scaler1 = Scaler(prune_rates[0])
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=None, track_running_stats=track)
        self.conv1 = nn.Conv2d(in_planes, new_conv1, kernel_size=3, stride=stride, padding=1, bias=False)

        self.scaler2 = Scaler(prune_rates[1])
        self.bn2 = nn.BatchNorm2d(new_conv1, momentum=None, track_running_stats=track)
        self.conv2 = nn.Conv2d(new_conv1, new_conv2, kernel_size=3, stride=1, padding=1, bias=False)

        # 修改点1：始终创建shortcut连接
        self.shortcut = nn.Conv2d(in_planes, new_conv2, kernel_size=1, stride=stride, bias=False)
        # 修改点2：记录是否需要使用shortcut
        self.need_shortcut = stride != 1 or in_planes != new_conv2

    def forward(self, x):
        out = F.relu(self.bn1(self.scaler1(x)))
        # 修改点3：根据need_shortcut决定是否使用shortcut
        shortcut = self.shortcut(out) if self.need_shortcut else x
        out = self.conv1(out)
        out = F.relu(self.bn2(self.scaler2(out)))
        out = self.conv2(out)
        out += shortcut
        return out

# -----------------------------
# Modified Bottleneck Block (Bottleneck)
# -----------------------------
class Bottleneck(nn.Module):
    """
    Modified Bottleneck block supporting per-layer prune rates.
    包含 3 个卷积层，其剪枝率列表为 [r1, r2, r3]:
      conv1: 使用 1×1 卷积，将 in_planes 映射为 new1 = ceil(planes * r1)
      conv2: 使用 3×3 卷积，将 new1 映射为 new2 = ceil(planes * r2)，步长为stride
      conv3: 使用 1×1 卷积，将 new2 映射为 new3 = ceil(planes * expansion * r3)
    最终输出通道数为 new3.
    
    修改点：始终创建shortcut连接，但在forward中根据need_shortcut决定是否使用。
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride, prune_rates, track):
        super().__init__()
        new1 = int(np.ceil(planes * prune_rates[0]))
        new2 = int(np.ceil(planes * prune_rates[1]))
        new3 = int(np.ceil(planes * self.expansion * prune_rates[2]))
        self.out_channels = new3

        self.scaler1 = Scaler(prune_rates[0])
        self.bn1 = nn.BatchNorm2d(in_planes, momentum=None, track_running_stats=track)
        self.conv1 = nn.Conv2d(in_planes, new1, kernel_size=1, bias=False)

        self.scaler2 = Scaler(prune_rates[1])
        self.bn2 = nn.BatchNorm2d(new1, momentum=None, track_running_stats=track)
        self.conv2 = nn.Conv2d(new1, new2, kernel_size=3, stride=stride, padding=1, bias=False)

        self.scaler3 = Scaler(prune_rates[2])
        self.bn3 = nn.BatchNorm2d(new2, momentum=None, track_running_stats=track)
        self.conv3 = nn.Conv2d(new2, new3, kernel_size=1, bias=False)

        # 修改点1：始终创建shortcut连接
        self.shortcut = nn.Conv2d(in_planes, new3, kernel_size=1, stride=stride, bias=False)
        # 修改点2：记录是否需要使用shortcut
        self.need_shortcut = stride != 1 or in_planes != new3

    def forward(self, x):
        out = F.relu(self.bn1(self.scaler1(x)))
        # 修改点3：根据need_shortcut决定是否使用shortcut
        shortcut = self.shortcut(out) if self.need_shortcut else x
        out = self.conv1(out)
        out = F.relu(self.bn2(self.scaler2(out)))
        out = self.conv2(out)
        out = F.relu(self.bn3(self.scaler3(out)))
        out = self.conv3(out)
        out += shortcut
        return out

# -----------------------------
# Modified ResNet
# -----------------------------
class ResNet(nn.Module):
    """
    Modified ResNet network supporting per-layer prune rates.
    参数 layer_prune_rates 为嵌套列表：
      - 外层列表对应 layer1 ~ layer4（大层）
      - 每个大层为列表，长度等于该层 block 数
      - 每个 block 又为一个剪枝率列表：
           对于基本块 Block，长度为2；
           对于 Bottleneck，长度为3。
    """
    def __init__(self, data_shape, hidden_size, block, num_blocks, num_classes, layer_prune_rates, track):
        super().__init__()
        self.in_planes = hidden_size[0]
        # 初始卷积层不作剪枝，保持固定通道数
        self.conv1 = nn.Conv2d(data_shape, hidden_size[0], kernel_size=3, stride=1, padding=1, bias=False)

        self.layer1 = self._make_layer(block, hidden_size[0], num_blocks[0],
                                       stride=1, layer_prune_rates=layer_prune_rates[0], track=track)
        self.layer2 = self._make_layer(block, hidden_size[1], num_blocks[1],
                                       stride=2, layer_prune_rates=layer_prune_rates[1], track=track)
        self.layer3 = self._make_layer(block, hidden_size[2], num_blocks[2],
                                       stride=2, layer_prune_rates=layer_prune_rates[2], track=track)
        self.layer4 = self._make_layer(block, hidden_size[3], num_blocks[3],
                                       stride=2, layer_prune_rates=layer_prune_rates[3], track=track)
        # 修改此处：bn4 和 linear 根据最后一层实际输出通道数来设置
        final_channels = self.in_planes
        self.bn4 = nn.BatchNorm2d(final_channels, momentum=None, track_running_stats=track)
        self.scaler = Scaler(1)  # 最后一阶段通常不缩放，可设为1
        self.linear = nn.Linear(final_channels, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride, layer_prune_rates, track):
        """
        layer_prune_rates: 列表，长度为 num_blocks，每个元素是当前 block 内的剪枝率列表
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i, s in enumerate(strides):
            prune_rates = layer_prune_rates[i]
            block_inst = block(self.in_planes, planes, s, prune_rates, track)
            layers.append(block_inst)
            # 更新 self.in_planes 为当前 block 输出通道数
            self.in_planes = block_inst.out_channels
        return nn.Sequential(*layers)

    def forward(self, x, kd=False):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.relu(self.bn4(self.scaler(out)))
        # if kd == True:
        #     out = {'features': out}
        #     return out
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out
    
    def __call__(self, x=None,kd = False):
        if x == None and kd == False:
            return self
        else:
            return self.forward(x,kd)
        

# -----------------------------
# 模型构造函数
# -----------------------------
def resnet18(layer_prune_rates=None, track=False,classes_size=10,all_rate=None):
    if all_rate != None:
        layer_prune_rates = [
        [[all_rate,all_rate], [all_rate,all_rate]],   # layer1
        [[all_rate,all_rate], [all_rate,all_rate]],   # layer2
        [[all_rate,all_rate], [all_rate,all_rate]],   # layer3
        [[all_rate,all_rate], [all_rate,all_rate]]    # layer4
    ]
    if layer_prune_rates == None:
        layer_prune_rates = [
        [[1,1], [1,1]],   # layer1
        [[1,1], [1,1]],   # layer2
        [[1,1], [1,1]],   # layer3
        [[1,1], [1,1]]    # layer4
    ]
    if Config().data.datasource == "CIFAR100":
        classes_size = 100

    data_shape = 3
    hidden_size = [64, 128, 256, 512]
    model = ResNet(data_shape, hidden_size, Block, [2, 2, 2, 2], classes_size, layer_prune_rates, track)
    model.apply(init_param)
    return model
def resnet18_CIFAR100(layer_prune_rates=None, track=False,classes_size=100,all_rate=None):
    if all_rate != None:
        layer_prune_rates = [
        [[all_rate,all_rate], [all_rate,all_rate]],   # layer1
        [[all_rate,all_rate], [all_rate,all_rate]],   # layer2
        [[all_rate,all_rate], [all_rate,all_rate]],   # layer3
        [[all_rate,all_rate], [all_rate,all_rate]]    # layer4
    ]
    if layer_prune_rates == None:
        layer_prune_rates = [
        [[1,1], [1,1]],   # layer1
        [[1,1], [1,1]],   # layer2
        [[1,1], [1,1]],   # layer3
        [[1,1], [1,1]]    # layer4
    ]
    if Config().data.datasource == "CIFAR100":
        classes_size = 100

    data_shape = 3
    hidden_size = [64, 128, 256, 512]
    model = ResNet(data_shape, hidden_size, Block, [2, 2, 2, 2], classes_size, layer_prune_rates, track)
    model.apply(init_param)
    return model
def resnet18_TinyImagenet(layer_prune_rates=None, track=False,classes_size=200,all_rate=None):
    if all_rate != None:
        layer_prune_rates = [
        [[all_rate,all_rate], [all_rate,all_rate]],   # layer1
        [[all_rate,all_rate], [all_rate,all_rate]],   # layer2
        [[all_rate,all_rate], [all_rate,all_rate]],   # layer3
        [[all_rate,all_rate], [all_rate,all_rate]]    # layer4
    ]
    if layer_prune_rates == None:
        layer_prune_rates = [
        [[1,1], [1,1]],   # layer1
        [[1,1], [1,1]],   # layer2
        [[1,1], [1,1]],   # layer3
        [[1,1], [1,1]]    # layer4
    ]
    if Config().data.datasource == "CIFAR100":
        classes_size = 100

    data_shape = 3
    hidden_size = [64, 128, 256, 512]
    model = ResNet(data_shape, hidden_size, Block, [2, 2, 2, 2], classes_size, layer_prune_rates, track)
    model.apply(init_param)
    return model

def resnet34(layer_prune_rates=None, track=False,classes_size=10,all_rate=None):
    if all_rate != None:
        layer_prune_rates = [
        [[all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate]],   # layer1
        [[all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate]],   # layer2
        [[all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate]],   # layer3
        [[all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate]]    # layer4
    ]
    if layer_prune_rates == None:
        layer_prune_rates = [
        [[1,1], [1,1], [1,1]],   # layer1
        [[1,1], [1,1], [1,1], [1,1]],   # layer2
        [[1,1], [1,1], [1,1], [1,1], [1,1], [1,1]],   # layer3
        [[1,1], [1,1], [1,1]]    # layer4
    ]
    """ResNet34 with per-layer prune rates, using basic blocks.
       对于 ResNet34，大层 block 数为 [3, 4, 6, 3].
    """
    if Config().data.datasource == "CIFAR100":
        classes_size = 100
    data_shape = 3
    hidden_size = [64, 128, 256, 512]
    model = ResNet(data_shape, hidden_size, Block, [3, 4, 6, 3], classes_size, layer_prune_rates, track)
    model.apply(init_param)
    return model
def resnet34_CIFAR100(layer_prune_rates=None, track=False,classes_size=100,all_rate=None):
    if all_rate != None:
        layer_prune_rates = [
        [[all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate]],   # layer1
        [[all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate]],   # layer2
        [[all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate]],   # layer3
        [[all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate]]    # layer4
    ]
    if layer_prune_rates == None:
        layer_prune_rates = [
        [[1,1], [1,1], [1,1]],   # layer1
        [[1,1], [1,1], [1,1], [1,1]],   # layer2
        [[1,1], [1,1], [1,1], [1,1], [1,1], [1,1]],   # layer3
        [[1,1], [1,1], [1,1]]    # layer4
    ]
    """ResNet34 with per-layer prune rates, using basic blocks.
       对于 ResNet34，大层 block 数为 [3, 4, 6, 3].
    """
    if Config().data.datasource == "CIFAR100":
        classes_size = 100
    data_shape = 3
    hidden_size = [64, 128, 256, 512]
    model = ResNet(data_shape, hidden_size, Block, [3, 4, 6, 3], classes_size, layer_prune_rates, track)
    model.apply(init_param)
    return model
def resnet34_TinyImagenet(layer_prune_rates=None, track=False,classes_size=200,all_rate=None):
    if all_rate != None:
        layer_prune_rates = [
        [[all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate]],   # layer1
        [[all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate]],   # layer2
        [[all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate]],   # layer3
        [[all_rate,all_rate], [all_rate,all_rate], [all_rate,all_rate]]    # layer4
    ]
    if layer_prune_rates == None:
        layer_prune_rates = [
        [[1,1], [1,1], [1,1]],   # layer1
        [[1,1], [1,1], [1,1], [1,1]],   # layer2
        [[1,1], [1,1], [1,1], [1,1], [1,1], [1,1]],   # layer3
        [[1,1], [1,1], [1,1]]    # layer4
    ]
    """ResNet34 with per-layer prune rates, using basic blocks.
       对于 ResNet34，大层 block 数为 [3, 4, 6, 3].
    """
    if Config().data.datasource == "CIFAR100":
        classes_size = 100
    data_shape = 3
    hidden_size = [64, 128, 256, 512]
    model = ResNet(data_shape, hidden_size, Block, [3, 4, 6, 3], classes_size, layer_prune_rates, track)
    model.apply(init_param)
    return model
def resnet50(layer_prune_rates, track=False,classes_size=10):
    """ResNet50 with per-layer prune rates, using Bottleneck blocks.
       对于 ResNet50，大层 block 数为 [3, 4, 6, 3]，每个 block 内剪枝率长度为 3.
    """
    if Config().data.datasource == "CIFAR100":
        classes_size = 100
    data_shape = 3
    hidden_size = [64, 128, 256, 512]
    model = ResNet(data_shape, hidden_size, Bottleneck, [3, 4, 6, 3], classes_size, layer_prune_rates, track)
    model.apply(init_param)
    return model

def resnet101(layer_prune_rates, track=False,classes_size=10):
    """ResNet101 with per-layer prune rates, using Bottleneck blocks.
       大层 block 数为 [3, 4, 23, 3].
    """
    if Config().data.datasource == "CIFAR100":
        classes_size = 100
    data_shape = 3
    hidden_size = [64, 128, 256, 512]
    model = ResNet(data_shape, hidden_size, Bottleneck, [3, 4, 23, 3], classes_size, layer_prune_rates, track)
    model.apply(init_param)
    return model

def resnet152(layer_prune_rates, track=False,classes_size=10):
    """ResNet152 with per-layer prune rates, using Bottleneck blocks.
       大层 block 数为 [3, 8, 36, 3].
    """
    if Config().data.datasource == "CIFAR100":
        classes_size = 100
    data_shape = 3
    hidden_size = [64, 128, 256, 512]
    model = ResNet(data_shape, hidden_size, Bottleneck, [3, 8, 36, 3], classes_size, layer_prune_rates, track)
    model.apply(init_param)
    return model

def count_parameters(model):
    "Count the number of trainable parameters in a model."
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



# -----------------------------
# 示例：创建与测试模型
# -----------------------------
if __name__ == "__main__":
    # 以 ResNet18 为例，定义每个 Block 内两个卷积层的剪枝率
    # layer_prune_rates = [
    #     [[0.9, 0.95], [0.92, 0.93]],         # layer1: 2 blocks
    #     [[0.88, 0.90], [0.87, 0.89]],         # layer2: 2 blocks
    #     [[0.85, 0.86], [0.84, 0.85]],         # layer3: 2 blocks
    #     [[0.80, 0.83], [0.79, 0.82]]          # layer4: 2 blocks
    # ]
    layer_prune_rates = [
        [[0.5, 0.5], [0.5, 0.5]],         # layer1: 2 blocks
        [[0.5, 0.5], [0.5, 0.5]],         # layer2: 2 blocks
        [[0.5, 0.5], [0.5, 0.5]],         # layer3: 2 blocks
        [[0.5, 0.5], [0.5, 0.5]],          # layer4: 2 blocks
    ]
    model = resnet18( track=False)
    print(model.state_dict().keys())
    input = torch.ones(3,3, 32, 32)
    x = model(input)
    print(model)
    print(f"Total parameters: {count_parameters(model)}")
