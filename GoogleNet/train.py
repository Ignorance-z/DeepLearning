# -*- coding: utf-8 -*-
import copy
import os

import matplotlib.pyplot as plt
import torch.cuda
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from net import GoogLeNet
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
    googleNet = GoogLeNet(1, 10, True, False).to(device)
    optimizer = optim.Adam(params=googleNet.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    # 计算版本一：
    best_acc = 0
    best_acc_list, best_loss_list = [], []
    for epoch in range(EPOCH):
        train_acc, train_loss = [], []
        googleNet.train()
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            trainOut, aux_out1, aux_out2 = googleNet(img)
            loss0 = criterion(trainOut, label)
            loss1 = criterion(aux_out1, label)
            loss2 = criterion(aux_out2, label)
            loss = loss0 + loss1*0.3 + loss2*0.3

            predict = torch.argmax(trainOut, dim=1)
            acc = (predict == label).sum() / label.shape[0]
            # print(train_len, label.shape[0], (predict == label).sum().item(), acc.item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss.append(loss.item())
            train_acc.append(acc.item())

        avg_loss = sum(train_loss) / len(train_loss)
        avg_acc = sum(train_acc) / len(train_acc)
        print(f'[{epoch+1}/{EPOCH}]: \ttrain_loss: {avg_loss}\ttrain_acc: {avg_acc}')

        # 计算版本二：
        # best_acc = 0
        # acc_list, loss_list = [], []
        # for epoch in range(EPOCH):
        #     train_loss, train_acc, n, batch_count = 0.0, 0.0, 0, 0
        #     googleNet.train()
        #     for img, label in train_loader:
        #         img, label = img.to(device), label.to(device)
        #         trainOut, aux_out1, aux_out2 = googleNet(img)
        #         loss0 = criterion(trainOut, label)
        #         loss1 = criterion(aux_out1, label)
        #         loss2 = criterion(aux_out2, label)
        #         loss = loss0 + loss1 * 0.3 + loss2 * 0.3
        #
        #         predict = torch.argmax(trainOut, dim=1)
        #
        #         optimizer.zero_grad()
        #         loss.backward()
        #         optimizer.step()
        #
        #         train_loss += loss.item()
        #         train_acc += (predict == label).sum().item()
        #         n += label.shape[0]
        #         batch_count += 1
        #
        #         avg_acc = train_acc / n
        #         avg_loss = train_loss / batch_count
        #
        #     print(f'[{epoch + 1}/{EPOCH}]: \ttrain_loss: {avg_loss}\ttrain_acc: {avg_acc}')

        if best_acc < avg_acc:
            best_acc = avg_acc
            best_loss_list = copy.deepcopy(train_loss)
            best_acc_list = copy.deepcopy(train_acc)
            torch.save(googleNet.state_dict(), os.path.join(SAVE_PATH, 'best_model.pth'))
        if epoch + 1 == EPOCH:
            torch.save(googleNet.state_dict(), os.path.join(SAVE_PATH, 'last_model.pth'))


# for i in range(1, 5):
#     plt.subplot(2, 2, i)
#     plt.imshow(data_train[i][0].transpose(0, 2))
#     plt.title(get_fashion_mnist_text_label(data_train[i][1]))
# plt.tight_layout()
# plt.show()