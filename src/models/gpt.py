# llms/gpt.py
from importlib import import_module
from openai import OpenAI
from loguru import logger
from src.models.base_llm import BaseLLM
conf = import_module("configs.config")


class GPT(BaseLLM):
    def __init__(self, model_name="gpt-4o-mini", temperature=None, max_new_tokens=None, report=False):
        super().__init__(model_name, temperature, max_new_tokens)
        self.report = report

        # 初始化 OpenAI 客户端
        self.client = OpenAI(
            api_key=conf.GPT_api_key,
            base_url=conf.GPT_api_base
        )
        if model_name is None:
            self.model_name = conf.GPT_model
        else:
            self.model_name = model_name
        if temperature is None:
            self.temperature = conf.GPT_temperature
        else:
            self.temperature = temperature

        if max_new_tokens is None:
            self.max_new_tokens = conf.max_new_tokens
        else:
            self.max_new_tokens = max_new_tokens

    def request(self, query: str, system_prompt: str = None) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": query})

        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_new_tokens,
        )

        res_text = resp.choices[0].message.content
        if self.report and hasattr(resp, "usage"):
            logger.info(
                f"[Claude] {self.model_name} token consumed: {resp.usage.total_tokens}"
            )

        return res_text