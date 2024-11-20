import os
from dotenv import load_dotenv
from llm_config import *

load_dotenv()


class QAAgent:
    """负责处理问题并调用GPT模型生成回答"""
    def __init__(self):
        if os.getenv("OLLAMA_MODEL"):
            self.client = OllamaService()
        elif os.getenv("OPENAI_MODEL"):
            self.client = OpenaiApiLlmService()
        elif os.getenv("OLLAMA_VISION_MODEL"):
            self.client = OllamaVisionService()

    def ask_gpt(self, messages, idx):
        success, idx, message, answer = call_gpt(
            messages, idx, self.client
        )
            
        return success, idx, message, answer
