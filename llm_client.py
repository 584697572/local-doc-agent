"""
大模型客户端初始化模块。

这个项目使用 OpenAI Python SDK 的兼容接口访问 DeepSeek。
API Key 不写在代码里，而是放在 .env 文件中，然后用 python-dotenv 加载。
"""

import os
from dotenv import load_dotenv
from openai import OpenAI

from config import BASE_URL

# 读取项目根目录下的 .env 文件。
# 例如 .env 中可以写：DEEPSEEK_API_KEY=sk-xxxx
load_dotenv()

# 创建一个全局 client，agent.py 直接 import 使用。
# os.getenv 会从环境变量中读取 DEEPSEEK_API_KEY。
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=BASE_URL,
)
