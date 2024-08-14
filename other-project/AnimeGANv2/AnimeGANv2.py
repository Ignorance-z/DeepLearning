import matplotlib.pyplot as plt
import torch
from PIL import Image
from model import Generator
from torchvision.transforms.functional import to_tensor, to_pil_image


def face2paint(
    img: Image,
    size: int,
    side_by_side: bool = True,
) -> Image.Image:

    model_fname = "face_paint_512_v2_0.pt"
    device = 'cpu'
    torch.set_grad_enabled(False)

    model = Generator().eval().to(device)
    model.load_state_dict(torch.load(model_fname))

    w, h = img.size
    s = min(w, h)
    # 裁剪，找到图片重点
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    # Image.LANCZOS高质量下采样滤波器
    img = img.resize((size, size), Image.LANCZOS)

    # *2-1用来调整亮度
    input = to_tensor(img).unsqueeze(0) * 2 - 1
    output = model(input.to(device)).cpu()[0]

    if side_by_side:
        output = torch.cat([input[0], output], dim=2)

    output = (output * 0.5 + 0.5).clip(0, 1)

    return to_pil_image(output)


# finish_deal = face2paint(pic, 512)
# plt.imshow(finish_deal)
# plt.axis('off')
# plt.savefig('pretty_girl.png', bbox_inches='tight', pad_inches=0)
# plt.show()



# pic = Image.open('./R-C.png')
# # print(pic_1.shape)
# img_tensor = transform(pic).unsqueeze(0)
# print(img_tensor.shape)
# # 网页获取
# # model = torch.hub.load("bryandlee/animegan2-pytorch", "generator").eval()
# out = model(img_tensor)  # BCHW tensor
# out = out.squeeze(0)
# out = out.permute(1, 2, 0).detach().numpy()
# # print(out.shape)
# plt.imshow(out)
# plt.show()

