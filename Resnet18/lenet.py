import  torch
from    torch import nn
from    torch.nn import functional as F
# from main import main1



class lenet(nn.Module):
    """
    for cifar10 dataset.
    """
    def __init__(self):
        super(lenet, self).__init__()
        self.conv_unit = nn.Sequential(
            nn.Conv2d(3, 6, 5, stride=1, padding=0),   # x: [b, 3, 32, 32] =>[b, 6, ]
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),

            nn.Conv2d(6, 16, 5, stride=1, padding=0),
            nn.AvgPool2d(kernel_size=2, stride=2, padding=0),
        )
        self.fc_unit = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10)
        )

        tmp = torch.randn(2, 3, 32, 32)
        out = self.conv_unit(tmp)
        print('conv out', out.shape)


        #
        self.criteon = nn.CrossEntropyLoss()

    def forward(self, x):


        batchsz= x.size(0)
        # [b, 3, 32, 32]  => [b,16, 5, 5]
        x = self.conv_unit(x)
        # [b, 16, 5, 5]   => [b, 16*5*5]
        x = x.view(batchsz, -1)
        #  [b, 16*5*5]  => [b, 10]
        logits = self.fc_unit(x)
        # logits= self.fc_unit(x)
        #  [b, 10]
        # pred =F.softmax(logits, dim=1)
        # loss = self.criteon(x, y)
        return logits




def main():
    # main1(x)

    net = lenet()
    tmp = torch.randn(2, 3, 32, 32)
    out = net(tmp)
    print('conv out', out.shape)










if __name__ == '__main__':
    main()