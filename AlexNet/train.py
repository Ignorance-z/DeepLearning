# -*- coding: utf-8 -*-
import os
import torch.cuda
import torchvision.models
from torch.utils.tensorboard import SummaryWriter
from net import AlexNet
from myDateset import MyData, transform
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import copy
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 32
LR = 0.01
EPOCH = 50
TRAIN_DIR_PATH = r'../data/dogs-vs-cats/train'
TEST_DIR_PATH = r'../data/dogs-vs-cats/test'
SAVE_PATH = r'./model'
LOG_PATH = r'./logs'

os.makedirs('./model', exist_ok=True)

if __name__ == '__main__':

    twrite = SummaryWriter(LOG_PATH)
    # print(device)
    alexnet = AlexNet(3, 2).to(device)
    train_dataset = MyData(TRAIN_DIR_PATH, transform=transform)
    # test_dataset = MyData(TEST_DIR_PATH, transform=transform)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    optimizer = optim.SGD(params=alexnet.parameters(), lr=LR)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    # 学习率每10轮变为原来的0.5
    # lr_scheduler = lr_scheduler.StepLR(optimizer=optimizer, step_size=10, gamma=0.5)

    best_loss_list = []
    best_acc_list = []

    best_acc = 0
    for epoch in range(EPOCH):
        # 满足条件更新学习率
        # lr_scheduler.step()
        train_loss_list = []
        train_acc_list = []
        alexnet.train()
        for batch, (imgs, classes) in enumerate(train_dataloader):
            imgs, classes = imgs.to(device), classes.to(device)
            train_output = alexnet(imgs)
            # print(type(train_output))
            # loss = F.cross_entropy(train_output, classes)
            loss = criterion(train_output, classes)

            predict = torch.argmax(train_output, dim=1)
            train_acc = (predict == classes).sum() / classes.shape[0]

            # train_acc_list.append(train_acc.detach().item())
            # train_loss_list.append(loss.detach().item())
            train_acc_list.append(train_acc.item())
            train_loss_list.append(loss.item())
            # print(f'第{batch}组： ', train_acc.item(), loss.item())
            # print('------------------------------')

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(train_acc.item(), loss.item())
            # print('------------------------------')

        avg_loss = sum(train_loss_list) / len(train_loss_list)
        avg_acc = sum(train_acc_list) / len(train_acc_list)
        twrite.add_scalar('Train.Loss', avg_loss, epoch)
        twrite.add_scalar('Train.Accuracy', avg_acc, epoch)

        # alexnet.eval()
        # test_loss_list = []
        # test_acc_list = []
        # loss = 0
        # for batch, (timgs, tclasses) in enumerate(test_dataloader):
        #     timgs, tclasses = timgs.to(device), tclasses.to(device)
        #     with torch.no_grad():
        #         test_output = alexnet(timgs)
        #         predict = torch.argmax(test_output, dim=1)
        #         loss = criterion(test_output, tclasses)
        #     test_acc = (predict == tclasses).sum() / tclasses.shape[0]
        #     # test_acc_list.append(test_acc.detach().item())
        #     # test_loss_list.append(loss.detach().item())
        #     test_acc_list.append(test_acc.item())
        #     test_loss_list.append(loss.item())
        #
        # avg_test_acc = sum(test_acc_list) / len(test_acc_list)
        # avg_test_loss = sum(test_loss_list) / len(test_loss_list)
        # twrite.add_scalar('Test.Accuracy', avg_test_acc, epoch)
        # twrite.add_scalar('Test.Loss', avg_test_loss, epoch)

        # print(f"[{epoch + 1}/{EPOCH}]\ttrain_Loss:  {avg_loss}\ttrain_acc:  {avg_acc}\t"
        #       f"test_acc:  {avg_test_acc}\ttest_loss:  {avg_test_loss}")
        print(f"[{epoch + 1}/{EPOCH}]\ttrain_Loss:  {avg_loss}\ttrain_acc:  {avg_acc}")

        if best_acc < avg_acc:
            best_acc = avg_acc
            best_loss_list = copy.deepcopy(train_loss_list)
            best_acc_list = copy.deepcopy(train_acc_list)
            torch.save(alexnet.state_dict(), os.path.join(SAVE_PATH, 'best_model.pkl'))
        if epoch + 1 == EPOCH:
            torch.save(alexnet.state_dict(), os.path.join(SAVE_PATH, 'last_model.pkl'))

    twrite.close()

    epoch_list = [i for i in range(EPOCH)]
    plt.plot(epoch_list, best_loss_list, label='Train Loss')
    plt.plot(epoch_list, best_acc_list, label='Train Accuracy')
    plt.legend()
    plt.xlabel('epoch')
    plt.ylabel('number')
    plt.title('Train line')
    plt.show()

