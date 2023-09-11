# -*- coding: utf-8 -*-
import torch.cuda
import torchvision.datasets
from matplotlib import pyplot as plt
from torchvision.transforms import transforms
from net import GoogLeNet


def get_fashion_mnist_text_label(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return text_labels[int(labels)]


test_data = torchvision.datasets.FashionMNIST(root='data',
                                              train=False, download=False,
                                              transform=transforms.Compose([
                                                  transforms.Resize((96, 96)),
                                                  transforms.ToTensor()
                                              ]))

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = GoogLeNet(1, 10, aux_logits=False).to(device)
missing_keys, unexpected_keys = model.load_state_dict(torch.load('./best_model.pth'), strict=False)
# model.load_state_dict(torch.load('./best_model.pth'), strict=False)

# 选择图片进行预测
pic = test_data[4][0]
# print(pic.shape)

plt.imshow(pic.transpose(0, 2))
plt.show()

pic = pic.to(device)

model.eval()
with torch.no_grad():
    output = model(pic.unsqueeze(0))
    # print(output)
    # print(torch.softmax(output, dim=1))
    predict = torch.argmax(torch.softmax(output, dim=1))
    # predict1 = torch.argmax(output)
    # print(predict)
    # print(predict1)
print(get_fashion_mnist_text_label(int(predict)))