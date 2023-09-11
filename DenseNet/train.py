# -*- coding: utf-8 -*-
import os

import torch
import torchvision
import torch.optim as optim
from matplotlib import pyplot as plt
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Subset


from DenseNet.net import DenseNet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 64
EPOCH = 20
LR = 0.01
SAVE_PATH = 'model'
LOG_PATH = 'log'

train_indices = torch.arange(0, 48000)
valid_indices = torch.arange(48000, 50000)
train_and_valid = torchvision.datasets.CIFAR10(root='./data', train=True,
                                               download=True, transform=transforms.ToTensor())
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                            download=True, transform=transforms.ToTensor())
train_dataset = Subset(train_and_valid, train_indices)
valid_dataset = Subset(train_and_valid, valid_indices)
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=True)

if __name__ == '__main__':
    dense = DenseNet().to(device)
    optimizer = optim.Adam(dense.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    acc_list, loss_list = [], []
    valid_acc_list, valid_loss_list = [], []
    best_acc = 0.0
    for epoch in range(EPOCH):
        train_loss,  train_acc = [], []
        dense.train()
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)
            train_out = dense(img)
            predict = torch.argmax(train_out, dim=1)

            loss = criterion(train_out, label)
            acc = (predict == label).sum() / label.shape[0]

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_acc.append(acc.item())
            train_loss.append(loss.item())

        avg_loss = sum(train_loss) / len(train_loss)
        avg_acc = sum(train_acc) / len(train_acc)
        print(f'[{epoch+1}/{EPOCH}]:\t train_loss：{avg_loss:.4f}，train_acc：{avg_acc:.4f}')
        acc_list.append(avg_acc)
        loss_list.append(avg_loss)

        dense.eval()
        valid_loss, valid_acc = [], []
        with torch.no_grad():
            for img, label in valid_loader:
                img, label = img.to(device), label.to(device)
                valid_out = dense(img)

                loss = criterion(valid_out, label)
                predict = torch.argmax(valid_out, dim=1)
                acc = (predict == label).sum() / label.shape[0]

                valid_acc.append(acc.item())
                valid_loss.append(loss.item())

            valid_avg_loss = sum(valid_loss) / len(valid_acc)
            valid_avg_acc = sum(valid_acc) / len(valid_acc)
            print(f'[{epoch+1}/{EPOCH}]:\t valid_loss：{valid_avg_loss:.4f}，valid_acc：{valid_avg_acc:.4f}')
            valid_acc_list.append(valid_avg_acc)
            valid_loss_list.append(valid_avg_loss)

        if best_acc < avg_acc:
            best_acc = avg_acc
            torch.save(dense.state_dict(), os.path.join(SAVE_PATH, 'best_model.pth'))
        if epoch + 1 == EPOCH:
            torch.save(dense.state_dict(), os.path.join(SAVE_PATH, 'final_model.pth'))


    epoch_list = [(i + 1) for i in range(EPOCH)]
    plt.figure()
    plt.plot(epoch_list, acc_list)
    plt.plot(epoch_list, valid_acc_list)
    plt.xlabel('epoch')
    plt.ylabel('percent')
    plt.legend(['train', 'valid'])
    plt.title('Acc')
    plt.show()
    plt.savefig('acc.png')

    plt.figure()
    plt.plot(epoch_list, loss_list)
    plt.plot(epoch_list, valid_loss_list)
    plt.xlabel('epoch')
    plt.ylabel('percent')
    plt.legend(['train', 'valid'])
    plt.title('Loss')
    plt.show()
    plt.savefig('loss.png')