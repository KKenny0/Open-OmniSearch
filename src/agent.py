import os
from dotenv import load_dotenv
from .llm_config import *

load_dotenv()


class QAAgent:
    """负责处理问题并调用GPT模型生成回答"""
    def __init__(self,
                 host="http://localhost:11434",
                 model="llama3.2-vision:11b-instruct-q4_K_M",
                 **kwargs):
        self.client = OllamaVisionService(host, model, **kwargs)

    def ask_gpt(self, messages, idx):
        success, idx, message, answer = call_gpt(
            messages, idx, self.client
        )
            
        return success, idx, message, answer
