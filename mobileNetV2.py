# mobileNetV2.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LinearBottleNeck(nn.Module):
    """
    完全仿照 mobileNetV2_yuan.py 的 LinearBottleNeck 结构，只是加上剪枝率 p 的计算。
    - in_channels: 该瓶颈输入通道
    - out_channels: 该瓶颈输出通道（投影后的通道）
    - stride: 卷积步长
    - t: 扩展倍数
    - prune_rates: [r1, r2, r3]，分别是 expansion、depthwise、projection 三个阶段的通道保留率
    - trs: track_running_stats，控制 BatchNorm 是否跟踪 running_mean / running_var
    """

    def __init__(self, in_channels, out_channels, stride, t, prune_rates, trs):
        super(LinearBottleNeck, self).__init__()

        # ========== 计算剪枝后的通道数 ==========
        # 1. expansion 第一层的输出通道 = ceil(in_channels * t * r1)
        expanded_channels = int(np.ceil(in_channels * t * prune_rates[0]))
        # 2. 第二层 depthwise 的输入输出通道 = expanded_channels * r2 ？ 
        #    但注意 depthwise 卷积要求输入输出通道一致，我们直接把它当作 "扩展后通道数"， 
        #    后面再用 batchnorm 变成实际拼接的通道。为了对齐原版行为，我们直接让 depthwise 
        #    的 in/out_channels = expanded_channels（因为原版没有在 depthwise 做剪枝）。
        depthwise_channels = expanded_channels
        # 3. projection 最后的输出通道 = ceil(out_channels * r3)
        projected_channels = int(np.ceil(out_channels * prune_rates[2]))

        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = projected_channels

        # ========== 仿照 Yuan 版，在一个 nn.Sequential 中依次构造 7 层 ==========
        layers = []

        # --- conv1: 1x1 expansion ---
        layers.append(nn.Conv2d(in_channels, expanded_channels, 1 ))
        layers.append(nn.BatchNorm2d(expanded_channels, track_running_stats=trs))
        layers.append(nn.ReLU6(inplace=True))

        # --- conv2: 3x3 depthwise ---
        layers.append(nn.Conv2d(expanded_channels, depthwise_channels, 3,
                                stride=stride, padding=1, groups=expanded_channels ))
        layers.append(nn.BatchNorm2d(depthwise_channels, track_running_stats=trs))
        layers.append(nn.ReLU6(inplace=True))

        # --- conv3: 1x1 projection ---
        layers.append(nn.Conv2d(depthwise_channels, projected_channels, 1 ))
        layers.append(nn.BatchNorm2d(projected_channels, track_running_stats=trs))

        # 把以上 7 层封装成一个 Sequential
        self.residual = nn.Sequential(*layers)

    def forward(self, x):
        out = self.residual(x)
        # 如果 stride=1 且 输入通道 == 输出通道，就做残差相加
        if self.stride == 1 and self.in_channels == self.out_channels:
            out = out + x
        return out


class MobileNetV2(nn.Module):
    """
    完全对齐 mobileNetV2_yuan.py 的整体架构，只是在初始化时支持传入各个 stage 的剪枝比率。
    - channels: 输入图片的通道数（一般是 3）
    - num_classes: 分类任务的类别数
    - stage_prune_rates: 七个 stage 各自的 prune_rates 列表，形状类似：
          [
            [[r1,r2,r3]],                          # stage0: 1 个 block
            [[r1,r2,r3], [r1,r2,r3]],             # stage1: 2 个 block
            [[r1,r2,r3], [r1,r2,r3], [r1,r2,r3]], # stage2: 3 个 block
            [[r1,r2,r3], [r1,r2,r3], [r1,r2,r3], [r1,r2,r3]], # stage3
            [[r1,r2,r3], [r1,r2,r3], [r1,r2,r3]], # stage4
            [[r1,r2,r3], [r1,r2,r3], [r1,r2,r3]], # stage5
            [[r1,r2,r3]]                           # stage6: 1 个 block
          ]
    - track_running_stats: BatchNorm 是否跟踪 running_mean / running_var
    """

    def __init__(self, channels=3, num_classes=10, stage_prune_rates=None, track_running_stats=True):
        super(MobileNetV2, self).__init__()

        # 如果没有传入剪枝比率，就把所有 prune_rate 设成 1.0 → 等于不剪枝
        if stage_prune_rates is None:
            stage_prune_rates = [
                [[1.0, 1.0 ]],                                        # stage0
                [[1.0, 1.0 ], [1.0, 1.0 ]],                     # stage1
                [[1.0, 1.0 ], [1.0, 1.0 ], [1.0, 1.0 ]],   # stage2
                [[1.0, 1.0 ], [1.0, 1.0 ], [1.0, 1.0 ], [1.0, 1.0 ]], # stage3
                [[1.0, 1.0 ], [1.0, 1.0 ], [1.0, 1.0 ]],   # stage4
                [[1.0, 1.0 ], [1.0, 1.0 ], [1.0, 1.0 ]],   # stage5
                [[1.0, 1.0 ]]                                        # stage6
            ]

        self.stage_prune_rates = stage_prune_rates

        # ================== 初始卷积层 ==================
        # 原版 Yuan 里第一层是 Conv2d(3, 32, 3, padding=1) + BN + ReLU6
        first_conv = nn.Sequential(
            nn.Conv2d(channels, 32, 3, padding=1 ),
            nn.BatchNorm2d(32, track_running_stats=track_running_stats),
            nn.ReLU6(inplace=True)
        )

        # 把所有的子模块先暂存在一个 list 里，最后再用 Sequential 串联
        features_list = [first_conv]

        # ================== 7 个 Stage 的配置 ==================
        # 每个 entry: [output_channels, num_blocks, stride_of_first_block, expand_ratio t]
        stage_configs = [
            [16, 1, 1, 1],   # stage0: 1 个 block，t=1
            [24, 2, 2, 6],   # stage1: 2 个 block，t=6
            [32, 3, 2, 6],   # stage2
            [64, 4, 2, 6],   # stage3
            [96, 3, 1, 6],   # stage4
            [160, 3, 2, 6],  # stage5
            [320, 1, 1, 6]   # stage6
        ]

        current_channels = 32  # 第一层输出是 32

        # 逐个创建每个 stage 的若干个 LinearBottleNeck
        for stage_idx, (out_channels, num_blocks, stride, t) in enumerate(stage_configs):
            # 拿出这个 stage 对应的 prune_rates 列表
            rates_for_stage = self.stage_prune_rates[stage_idx]

            # 一共要做 num_blocks 个瓶颈模块
            blocks = []
            for block_idx in range(num_blocks):
                block_stride = stride if block_idx == 0 else 1
                # 这一块的三段剪枝率
                prune_rate_block = rates_for_stage[block_idx]
                # 构造 LinearBottleNeck 模块
                bottleneck = LinearBottleNeck(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    stride=block_stride,
                    t=t,
                    prune_rates=prune_rate_block,
                    trs=track_running_stats
                )
                blocks.append(bottleneck)
                # 更新 current_channels 为本瓶颈的输出通道
                current_channels = bottleneck.out_channels

            # 把这一整个 stage（若干个 block）打包成一个 Sequential
            features_list.append(nn.Sequential(*blocks))

        # ================== 最后一个 Conv 层 (1x1→1280) ==================
        # 跟 Yuan 版完全一样：Conv2d(current_channels, 1280, 1) + BN + ReLU6
        final_conv = nn.Sequential(
            nn.Conv2d(current_channels, int(np.ceil(1280 * 1.0)), 1 ),
            nn.BatchNorm2d(int(np.ceil(1280 * 1.0)), track_running_stats=track_running_stats),
            nn.ReLU6(inplace=True)
        )
        features_list.append(final_conv)

        # ========== 把上述所有模块合并成 self.features = Sequential(...) ==========
        self.features = nn.Sequential(*features_list)

        # ================== classifier (1x1 conv → num_classes) ==================
        # Yuan 版也是用 Conv2d(1280, num_classes, 1) 而不是 Linear
        self.classifier = nn.Conv2d(int(np.ceil(1280 * 1.0)), num_classes, 1)

    def forward(self, x):
        # 1. 先走所有特征层
        x = self.features(x)

        # 2. 全局池化到 1×1
        x = F.adaptive_max_pool2d(x, 1)

        # 3. “representation”：全局池化展平
        result = {'representation': x.view(x.size(0), -1)}

        # 4. classifier conv → flatten
        x = self.classifier(x)
        x = x.view(x.size(0), -1)
        result['output'] = x
        return result


def count_parameters(model):
    """
    计算模型可训练参数总数
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def mobilenetv2(stage_prune_rates=None, track_running_stats=False, num_classes=10):
    """
    便捷接口，根据传入的剪枝比率构造一个 MobileNetV2
    """
    return MobileNetV2(
        channels=3,
        num_classes=num_classes,
        stage_prune_rates=stage_prune_rates,
        track_running_stats=track_running_stats
    )


if __name__ == '__main__':
    # 简单演示：加载一个无剪枝模型并打印它的结构，确认名称对齐
    print("=" * 60)
    print("MobileNetV2 (剪枝对齐) 模型结构检查")
    print("=" * 60)

    # 全 1.0 意味不做任何剪枝
    default_rates = [
        [[1.0, 1.0 ]],                                 # stage0
        [[1.0, 1.0 ], [1.0, 1.0 ]],                # stage1
        [[1.0, 1.0 ], [1.0, 1.0 ], [1.0, 1.0 ]],  # stage2
        [[1.0, 1.0 ], [1.0, 1.0 ], [1.0, 1.0 ], [1.0, 1.0 ]],  # stage3
        [[1.0, 1.0 ], [1.0, 1.0 ], [1.0, 1.0 ]],  # stage4
        [[1.0, 1.0 ], [1.0, 1.0 ], [1.0, 1.0 ]],  # stage5
        [[1.0, 1.0 ]]                                    # stage6
    ]

    model = mobilenetv2(
        stage_prune_rates=default_rates,
        track_running_stats=False,
        num_classes=10
    )
    print(model)
    print(model.state_dict().keys())
    print(f"参数总量: {count_parameters(model):,}")
