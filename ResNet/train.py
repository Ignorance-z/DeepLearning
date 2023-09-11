# -*- coding: utf-8 -*-
import copy
import os

import matplotlib.pyplot as plt
from utils import Block, BottleNeck
import torch.cuda
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from net import ResNet
import torch.optim as optim

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 64
EPOCH = 30
LR = 0.01
SAVE_PATH = 'model'
LOG_PATH = 'log'


data_train = torchvision.datasets.FashionMNIST(root='data',
                                               train=True, download=True,
                                               transform=transforms.Compose([
                                                   transforms.Resize((96, 96)),
                                                   transforms.ToTensor()
                                               ]))
data_test = torchvision.datasets.FashionMNIST(root='data',
                                              train=False, download=False,
                                              transform=transforms.Compose([
                                                  transforms.Resize((96, 96)),
                                                  transforms.ToTensor()
                                              ]))
train_loader = torch.utils.data.DataLoader(dataset=data_train, batch_size=BATCH_SIZE, shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=BATCH_SIZE, shuffle=True)
train_len = len(data_train)
test_len = len(data_test)

if __name__ == '__main__':
    resNet = ResNet(1, BottleNeck, [3, 4, 6, 3]).to(device)
    optimizer = optim.SGD(params=resNet.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    acc_list, loss_list = [], []
    for epoch in range(EPOCH):
        train_loss, train_acc, n, batch_count = 0.0, 0.0, 0, 0
        resNet.train()
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            trainOut = resNet(img)
            loss = criterion(trainOut, label)
            predict = torch.argmax(trainOut, dim=1)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += (predict == label).sum().item()
            n += label.shape[0]
            batch_count += 1

        avg_acc = train_acc / n
        avg_loss = train_loss / batch_count
        acc_list.append(avg_acc)
        loss_list.append(avg_loss)

        print(f'[{epoch + 1}/{EPOCH}]: \ttrain_loss: {avg_loss}\ttrain_acc: {avg_acc}')

        if best_acc < avg_acc:
            best_acc = avg_acc
            torch.save(resNet.state_dict(), os.path.join(SAVE_PATH, 'best_model.pth'))
        if epoch + 1 == EPOCH:
            torch.save(resNet.state_dict(), os.path.join(SAVE_PATH, 'last_model.pth'))

    epoch_list = [(i + 1) for i in range(EPOCH)]
    plt.plot(epoch_list, acc_list)
    plt.plot(epoch_list, loss_list)
    plt.xlabel('epoch')
    plt.ylabel('per')
    plt.legend(['acc', 'loss'])
    plt.show()
    plt.savefig('best.png')

# for i in range(1, 5):
#     plt.subplot(2, 2, i)
#     plt.imshow(data_train[i][0].transpose(0, 2))
#     plt.title(get_fashion_mnist_text_label(data_train[i][1]))
# plt.tight_layout()
# plt.show()