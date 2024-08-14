# -*- coding: utf-8 -*-
import copy
import os
import torch.cuda
from torch import nn
from net import VGG
from dataset import MyData
from utils import second_transform
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
import torch.optim as optim
from utils import make_valid


# 完成数据集的重新划分
# make_valid()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

BATCH_SIZE = 64
LR = 0.01
EPOCH = 30
TRAIN_PATH = r'../data/dogs-vs-cats/train'
VALID_PATH = r'../data/dogs-vs-cats/valid'
TEST_PATH = r'../data/dogs-vs-cats/test'
MODEL_SAVE = r'./model'
LOG_PATH = r'./logs'
os.makedirs(MODEL_SAVE, exist_ok=True)

if __name__ == '__main__':
    log_write = SummaryWriter(LOG_PATH)
    layers = int(input('请输入VGG层数：16 or 19？'))
    vgg = VGG(layers_num=layers, in_channels=3).to(device)

    trainSet = MyData(TRAIN_PATH, transform=second_transform)
    trainLoader = DataLoader(trainSet, batch_size=BATCH_SIZE, shuffle=True)
    validSet = MyData(VALID_PATH, transform=second_transform)
    validLoader = DataLoader(validSet, batch_size=BATCH_SIZE, shuffle=True)
    # 定义激活函数
    optimizer = optim.SGD(params=vgg.parameters(), lr=LR)
    # 定义损失函数
    criterion = nn.CrossEntropyLoss()

    best_acc = 0
    best_acc_list, best_loss_list = [], []
    for epoch in range(EPOCH):
        trainLoss, trainAcc = [], []
        vgg.train()
        for img, label in trainLoader:
            img, label = img.to(device), label.to(device)
            trainOutput = vgg(img)
            loss = criterion(trainOutput, label)

            predict = torch.argmax(trainOutput, dim=1)
            acc = (predict == label).sum() / label.shape[0]

            trainLoss.append(loss.detach().item())
            trainAcc.append(acc.detach().item())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = sum(trainLoss) / len(trainLoss)
        avg_acc = sum(trainAcc) / len(trainAcc)
        log_write.add_scalar('Train.Loss', avg_loss, epoch)
        log_write.add_scalar('Train.Accuracy', avg_acc, epoch)

        # 在验证集上进行检验，不进行权重更新
        vgg.eval()
        validLoss, validAcc = [], []
        with torch.set_grad_enabled(False):
            for img_v, label_v in validLoader:
                img_v, label_v = img_v.to(device), label_v.to(device)
                validOutput = vgg(img_v)
                loss_valid = criterion(validOutput, label_v)
                predict_valid = torch.argmax(validOutput, dim=1)
                acc_valid = (predict_valid == label_v).sum() / label_v.shape[0]

                validLoss.append(loss_valid.detach().item())
                validAcc.append(acc_valid.detach().item())

            avg_acc_v = sum(validAcc) / len(validAcc)
            avg_loss_v = sum(validLoss) / len(validLoss)
            log_write.add_scalar('Valid.Accuracy', avg_acc_v, epoch)
            log_write.add_scalar('Valid.Loss', avg_loss_v, epoch)

        print(f"[{epoch + 1}/{EPOCH}]\ttrain_Loss:  {avg_loss}\ttrain_acc:  {avg_acc}\t"
              f"valid_acc:  {avg_acc_v}\tvalid_loss:  {avg_loss_v}")

        if best_acc < avg_acc:
            best_acc = avg_acc
            best_loss_list = copy.deepcopy(trainLoss)
            best_acc_list = copy.deepcopy(trainAcc)
            torch.save(vgg.state_dict(), os.path.join(MODEL_SAVE, '/kaggle/working/best_model.pth'))
        if epoch + 1 == EPOCH:
            torch.save(vgg.state_dict(), os.path.join(MODEL_SAVE, '/kaggle/working/last_model.pth'))

    log_write.close()