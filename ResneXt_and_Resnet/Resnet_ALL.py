import torch.nn as nn
import torch
from torchvision.models import T






class ResNet(nn.Module):

    def __init__(self, block, blocks_num, numclass=1000, include_top = True):
        super(ResNet, self).__init__()

        self.include_top = include_top
        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self.make_layer(block, 64, blocks_num[0])
        self.layer2 = self.make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self.make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self.make_layer(block, 512, blocks_num[3], stride=2)

        if include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(512*block.expansion, numclass)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')



    def make_layer(self, block, channel, block_num, stride=1):              #######   一个layer对应一个残差结构
        Need_conv = None
        if stride != 1 or self.in_channel !=channel * block.expansion:
            Need_conv = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))
        layers = []
        layers.append(block(self.in_channel, channel, Need_conv=Need_conv, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)

        return x



def resnet18(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


# def resnext50_32x4d(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
#     groups = 32
#     width_per_group = 4
#     return ResNet(Bottleneck, [3, 4, 6, 3],
#                   num_classes=num_classes,
#                   include_top=include_top,
#                   groups=groups,
#                   width_per_group=width_per_group)
#
#
# def resnext101_32x8d(num_classes=1000, include_top=True):
#     # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
#     groups = 32
#     width_per_group = 8
#     return ResNet(Bottleneck, [3, 4, 23, 3],
#                   num_classes=num_classes,
#                   include_top=include_top,
#                   groups=groups,
#                   width_per_group=width_per_group)



























###############################################             50 101 152 残差结构！！！！！！！！！！！！
class Bottleneck(nn.Module):
    """
    注意：原论文中，在虚线残差结构的主分支上，第一个1x1卷积层的步距是2，第二个3x3卷积层步距是1。
    但在pytorch官方实现过程中是第一个1x1卷积层的步距是1，第二个3x3卷积层步距是2，
    这么做的好处是能够在top1上提升大概0.5%的准确率。
    可参考Resnet v1.5 https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch
    """
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, Need_conv=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        # ------------------------------------------------------------
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # ------------------------------------------------------------
        self.conv3 = nn.Conv2d(out_channel, out_channel*self.expansion, kernel_size=1, stride=1)
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.Need_conv= Need_conv

    def forward(self, x):
        origin = x
        if self.Need_conv is not None:
            origin = self.Need_conv(x)
        out = self.conv1(x)
        out = self.relu(self.bn1(out))

        out = self.conv2(out)
        out = self.relu(self.bn2(out))

        out = self.conv3(out)
        out = self.bn3(out)

        out +=origin
        out = self.relu(out)
        return out







#######################################   18  34层残差结构！！！！！！！！！！
class BasicBlock(nn.Module):

    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, Need_conv=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.ADD = Need_conv

    def forward(self, x):

        origin = x
        if self.ADD is not None:
            origin = self.ADD(x)

        out = self.conv1(x)
        out = self.relu(self.bn1(out))
        out = self.conv2(out)
        out = self.bn2(out)

        out = out+origin
        out = self.relu(out)

        return out

























