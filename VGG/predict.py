# -*- coding: utf-8 -*-
import torch
from net import VGG
from torchvision import transforms
from PIL import Image


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

device = "cuda" if torch.cuda.is_available() else "cpu"

model = VGG(16, 3).to(device)
# torch.load(model1.parameters(), "model1/best_model.pth")
model.load_state_dict(torch.load('./best_model.pth'))
# model1 = torch.load('model1/best_model.pth')

pic_root = r"../data/dogs-vs-cats/test/3.jpg"

pic = Image.open(pic_root)
pic = transform(pic).to(device)

model.eval()
with torch.no_grad():
    output = model(pic.unsqueeze(0))
    predict = torch.argmax(output, dim=1)
    if predict == 1:
        print("dog")
    else:
        print("cat")

# output = model(pic.unsqueeze(0))
# predict = torch.argmax(output, dim=1)
# if predict == 1:
#     print("dog")
# else:
#     print("cat")