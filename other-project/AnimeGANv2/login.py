# -*- coding: utf-8 -*-
import streamlit as st


def login():
    st.title("用户登录")
    username = st.text_input("用户名")
    password = st.text_input("密码", type="password")
    login_button = st.button("登录")

    if login_button:
        if username == "admin" and password == "password":
            st.success("登录成功！")
        else:
            st.error("用户名或密码错误！")

login()