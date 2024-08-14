# -*- coding: utf-8 -*-
import streamlit as st
import threading
import time


def long_running_task():
    # 模拟一个长时间运行的任务
    for i in range(10):
        time.sleep(1)
        st.text("Task running...")

    st.text("Task complete!")


# Streamlit应用程序
def main():
    st.title("异步任务示例")

    if st.button("启动任务"):
        # 创建并启动一个新线程来执行任务
        thread = threading.Thread(target=long_running_task)
        thread.start()

    st.text("等待任务完成...")


# 运行Streamlit应用程序
if __name__ == '__main__':
    main()