# -*- coding: utf-8 -*-
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt

class MyData(Dataset):
    def __init__(self, dir_path, transform=None):
        self.dir_path = dir_path
        self.name_path = os.listdir(dir_path)
        self.transform = transform

    def __getitem__(self, idx):
        label = self.name_path[idx].split('.')[0].strip('')
        pic_path = os.path.join(self.dir_path, self.name_path[idx])
        pic = Image.open(pic_path)
        if self.transform is not None:
            pic = self.transform(pic)
        # 规定0是cat，1是dog
        return pic, 0 if label == 'cat' else 1

    def __len__(self):
        return len(self.name_path)


transform = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])


# root = r"../data/dogs-vs-cats/train"
# dst = MyData(root, transform=transform)
# for i in range(1, 5):
#     plt.subplot(2, 2, i)
#     # print(type(dst[i][0]))
#     # print(type(dst[i][0].transpose(0, 2)))
#     # dst[i][0].transpose(0, 2)
#     # print(dst[i][0].transpose(0, 2).shape)
#     plt.imshow(dst[i][0].transpose(0, 2))
#     plt.title(dst[i][1])
# plt.tight_layout()
# plt.show()