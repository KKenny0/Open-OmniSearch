import io
import traceback
from pathlib import Path
import os
from typing import Tuple

import requests
import time
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

workers = 4
skip_lines = 0

LIMIT = 1000000000000
retry_attempt = 10


__all__ = ["OpenaiApiLlmService",
           "OllamaService",
           "OllamaVisionService",
           "call_gpt"]


class OpenaiApiLlmService:
    def __init__(self,
                 model: str = "gpt-4o",
                 api_key: str = ""):
        if model_name := os.getenv("OPENAI_MODEL", ""):
            self.model = model_name
        else:
            self.model = model

        if api_key_env := os.getenv("OPENAI_API_KEY", ""):
            self.api_key = api_key_env
        else:
            self.api_key = api_key

    def __call__(self, messages, idx):
        """
        调用 gpt 的 api 回答问题，包含了违规检查和内容过滤错误。
        但不是官方的 api，其中的一些参数可能要修改成 openai 的官方参数规范
        """
        # 准备请求数据，包括模型、对话信息等参数
        data = {
            "model": self.model,
            "messages": messages,
            "n": 1,  # 回答数量
            "max_tokens": 4096
        }

        answer = None
        while answer is None:
            try:
                r = requests.post(
                    'https://api.openai.com/v1/chat/completions',
                    json=data,
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                resp = r.json()

                # 检查 API 响应的状态码与内容
                if r.status_code != 200:
                    print("请求失败，重试中！")
                    print(resp)
                    continue

                # 检查内容策略的相关代码
                if 'choices' in resp and resp['choices'][0].get('finish_reason') in ['content_filter',
                                                                                     'ResponsibleAIPolicyViolation']:
                    print('内容不符合策略要求，返回空结果')
                    return False, idx, "", "", 0, 0, 0

                message = resp['choices'][0]['message']
                answer = message['content']

                return True, idx, message, answer

            except Exception as e:
                logger.error(e)
                logger.error('发生异常，重试中！')
                time.sleep(1)  # 等待一段时间再重试
                continue


class OllamaService:
    def __init__(self,
                 host: str = "http://localhost:11434",
                 model: str = "qwen2.5:14b-instruct"):
        try:
            import ollama
        except ImportError:
            raise ValueError(
                "The ollama python package is not installed. "
                "Please install it with `pip install ollama`"
            )

        if ollama_host := os.getenv("OLLAMA_HOST", ""):
            self.host = ollama_host
        else:
            self.host = host

        if ollama_model := os.getenv("OLLAMA_MODEL", ""):
            self.model = ollama_model
        else:
            self.model = model

        self._client = ollama.Client(self.host)

    def __call__(self, messages, idx):
        answer = None
        while answer is None:
            try:
                response = self._client.chat(
                    messages=messages,
                    model=self.model,
                )

                message = response['message']
                answer = message['content']

                return True, idx, messages, answer

            except Exception as e:
                logger.error(e)
                logger.error('发生异常，重试中！')
                time.sleep(1)  # 等待一段时间再重试
                continue


class OllamaVisionService:
    def __init__(self,
                 host: str = "http://localhost:11434",
                 model: str = "llama3.2-vision",
                 **kwargs):
        try:
            import ollama
        except ImportError:
            raise ValueError(
                "The ollama python package is not installed. "
                "Please install it with `pip install ollama`"
            )

        if ollama_host := os.getenv("OLLAMA_HOST", ""):
            self.host = ollama_host
        else:
            self.host = host

        if ollama_model := os.getenv("OLLAMA_VISION_MODEL", ""):
            self.model = ollama_model
        else:
            self.model = model

        self._client = ollama.Client(self.host)
        self.args = kwargs

    def __call__(self, messages, idx):
        answer = None
        for idx, msg in enumerate(messages):
            if msg.get("images", None) and isinstance(msg["images"][0], io.BytesIO):
                file_path = Path("temp.png")
                with file_path.open("wb") as file:
                    file.write(msg["images"][0].getvalue())
                messages[idx].update({"images": [file_path]})

        while answer is None:
            try:
                response = self._client.chat(
                    messages=messages,
                    model=self.model,
                    options=self.args,
                )
                message = response['message']
                answer = message['content']
                return True, idx, message, answer

            except Exception as e:
                logger.error(e)
                logger.error(traceback.format_exc())
                logger.error('发生异常，重试中！')
                time.sleep(1)  # 等待一段时间再重试
                continue


def call_gpt(messages, idx, llm_server) -> Tuple[bool, int, str, str]:
    """
    调用 gpt 的 api 回答问题，包含了违规检查和内容过滤错误。
    但不是官方的 api，其中的一些参数可能要修改成 openai 的官方参数规范
    """
    return llm_server(messages, idx)
