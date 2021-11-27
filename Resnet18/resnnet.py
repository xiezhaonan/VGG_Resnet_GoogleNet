import torch
from torch import nn
from torch.nn import functional as F


class  Resblk(nn. Module):

    def __init__(self, ch_in, ch_out, stride=1):
        super(Resblk, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(ch_out)
        self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(ch_out)

        self.extra = nn.Sequential()
        if ch_out != ch_in:
            ####
             self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=stride,),
                nn.BatchNorm2d(ch_out)
            )

    def forward(self, x):
        # print('x.shape', x.shape)
        out = F.relu(self.bn1(self.conv1(x)))
        # print('out1.shape', out.shape)
        out = self.bn2(self.conv2(out))
        # print('out2.shape', out.shape)
        ######  short cut.
        #######  element_wise add
        # print('123', self.extra(x).shape, out.shape)
        out = self.extra(x) + out
        # print('out2', x.shape)
        out = F.relu(out)

        return out



class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )
        ######## followed 4 blocks
        ######## [b, 64, h, w]  => [b, 128, h, w]
        self.blk1 = Resblk(64, 128, stride=2)
        ######## [b, 128, h, w] => [b, 256, h, w]
        self.blk2 = Resblk(128, 256, stride=2)
        ########
        self.blk3 = Resblk(256, 512, stride=2)
        self.blk4 = Resblk(512, 512, stride=2)

        self.out = nn.Linear(512*1*1, 10)

    def forward(self, x):

        x = F.relu(self.conv1(x))
        ###### [b,64, h, w]  => [b, 1024. h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        x = F.adaptive_avg_pool2d(x, [1, 1])
        x = x.view(x.size(0), -1)
        x = self.out(x)

        return x

def main():
    # blk = Resblk(64, 128, stride=4)
    # tmp = torch.randn(2, 64, 32, 32)
    # out = blk(tmp)
    # print('block:',out.shape)
    #
    # x = torch.randn(2, 3, 32, 32)
    # model = Resnet18()
    # out = model(x)
    # print('resnet:', out.shape)
    blk = Resblk(64, 128, stride=4)
    tmp = torch.randn(2, 64, 32, 32)
    out = blk(tmp)
    print('block:', out.shape)

    x = torch.randn(512, 3, 32, 32)
    model = Resnet18()
    out = model(x)
    print('resnet:', out.shape)


if __name__ == '__main__':
    main()