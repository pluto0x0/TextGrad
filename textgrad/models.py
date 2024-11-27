import os
from typing import Dict, Any
from .main import Variable, BaseLLM


class OpenAI(BaseLLM):
    def __init__(self, key=None, model="gpt-4o-mini"):
        super().__init__()
        if key is None:
            key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise Exception("API key not found")

        from openai import OpenAI
        self.client = OpenAI(
            api_key=key,
        )
        self.model = model

    def get_response(self, msg):
        if isinstance(msg, str):
            msg = [
                {
                    "role": "user",
                    "content": msg,
                }
            ]
        chat_completion = self.client.chat.completions.create(
            messages=msg,
            model=self.model,
        )
        return chat_completion.choices[0].message.content
