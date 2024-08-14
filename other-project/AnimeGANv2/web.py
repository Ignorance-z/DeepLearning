# -*- coding: utf-8 -*-
import numpy as np
import streamlit as st
from PIL import Image
from AnimeGANv2 import face2paint
import base64
from io import BytesIO
import time


def image_to_base64(image):
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()


def sub_title():
    st.set_page_config(page_title="图像生成", page_icon=' ', layout="wide")
    st.title('图片动漫化')
    st.write('基于AnimeGANv2实现')
    st.write('建议上传实景图片，建议图片大小512*512，人像为佳。')


sub_title()
uploaded_file = st.file_uploader("上传一张图片", type=['png', 'jpg', 'jpeg'])
if uploaded_file:
    t = time.localtime()
    img = Image.open(uploaded_file).convert('RGB')
    # st.image(np.array(img), channels="RGB")
    deal_img = face2paint(img, 512, False)
    # 使用CSS来设置图像样式
    st.markdown(
        """
        <style>
        .centered-image {
            display: flex;
            justify-content: center;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # 在居中展示的容器中显示图像对象
    st.markdown(
        """
        <div>
            <span>原图：</span>
        </div>
        <div class="centered-image">
            <img src="data:image/png;base64,{}" alt="Image" width="512">
        </div>
        """.format(image_to_base64(img)),
        unsafe_allow_html=True
    )
    st.write('----')
    st.markdown(
        """
        <div>
            <span>动漫化后：</span>
        </div>
        <div class="centered-image">
            <img src="data:image/png;base64,{}" alt="Image" width="512">
        </div>
        """.format(image_to_base64(deal_img)),
        unsafe_allow_html=True
    )

    photo_name = 'photo/img_{}_{}_{}.png'.format(t.tm_hour, t.tm_min, t.tm_sec)
    deal_img.save(photo_name)
    with open(photo_name, "rb") as f:
        btn = st.download_button(
            label="Download image",
            data=f,
            file_name=photo_name,
            mime="image/png"
        )
    st.write('文件名：%s' % photo_name)
    # st.image(np.array(deal_img), channels="RGB")