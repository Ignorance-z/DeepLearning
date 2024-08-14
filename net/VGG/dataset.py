# -*- coding: utf-8 -*-
import os
import torch
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from utils import first_transform


class MyData(Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.name_list = os.listdir(self.dir_path)
        self.transform = transform

    def __getitem__(self, idx):
        label = self.name_list[idx].split('.')[0].strip('')
        img_path = os.path.join(self.dir_path, self.name_list[idx])
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        return img, 0 if label == 'cat' else 1

    def __len__(self):
        return len(self.name_list)


# train_set = MyData(r'../data/dogs-vs-cats/train', first_transform)
# train_loader = DataLoader(train_set, batch_size=1, shuffle=False)
# mean = torch.zeros(3)
# std = torch.zeros(3)
# for img, _ in train_loader:
# # #     print(img)
# # #     print('------------------\n', _)
#     for d in range(3):
#         mean[d] += img[:, d, :, :].mean()
#         std[d] += img[:, d, :, :].std()
# mean.div_(len(train_set))
# std.div_(len(train_set))
# print(mean, std)