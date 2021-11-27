import  torch
from torch.utils.data import DataLoader
from torchvision import  datasets
from torchvision import  transforms
from lenet import lenet
from resnnet import Resblk, Resnet18
from torch import nn, optim

def main():
    batchsz=128
    cifar_train = datasets.CIFAR10('cifar',train=True, download=True, transform=transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor()
        ]))
    cifar_train = DataLoader(cifar_train, batch_size=batchsz, shuffle=True)

    cifar_test = datasets.CIFAR10('cifar', train=False, download=False, transform=transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor()
    ]))
    cifar_test = DataLoader(cifar_test, batch_size=batchsz, shuffle=True)

    x, label = iter(cifar_train).next()
    print('x:',x.shape,'label:',label.shape)

    device = torch.device('cuda')
    # model = lenet().to(device)

    model = Resnet18().to(device)

    criteon = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(model)

#########################     Train!!!!
    for epoch in range(100):

        model.train()
        for batchidx, (x, label) in enumerate(cifar_train):
            x, label = x.to(device), label.to(device)
            logits = model(x)
            #  logits[b, 10]
            #  label[b]
            loss = criteon(logits, label)


            #####  backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        #######
        print('epoch:',epoch, 'loss:',loss.item())


        model.eval()
    # with torch.no_grad():
        total_correct = 0
        total_sum = 0
##################################   Test!!!!
        for x, label in cifar_test:
            x, label = x.to(device), label.to(device)
            ########  [b, 10]
            logits = model(x)
            #####   [b]
            pred = logits.argmax(dim=1)
            ##### [b] vs [b]  => scalar tensor
            total_correct += torch.eq(pred, label).float().sum()
            total_sum += x.size(0)
        acc = total_correct / total_sum
        print(epoch, 'test acc:',acc)


if __name__ == '__main__':
    main()