import os
from dotenv import load_dotenv
from openai import OpenAI

from config import BASE_URL

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url=BASE_URL,
)