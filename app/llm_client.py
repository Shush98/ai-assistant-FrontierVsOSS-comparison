import time
from openai import OpenAI
from gradio_client import Client
from app import config


class LLMClient:
    """One interface, two providers: 'frontier' (OpenAI) or 'oss' (HF Space)."""

    def __init__(self, provider: str):
        if provider not in ("frontier", "oss"):
            raise ValueError(f"unknown provider: {provider}")
        self.provider = provider
        if provider == "frontier":
            self._openai = OpenAI(api_key=config.OPENAI_API_KEY)
        else:
            self._gradio = Client(config.HF_SPACE_URL)

    def chat(self, messages: list[dict]) -> dict:
        """messages = [{'role':..., 'content':...}]. Returns normalized dict."""
        if self.provider == "frontier":
            return self._chat_frontier(messages)
        return self._chat_oss(messages)

    def _chat_frontier(self, messages: list[dict]) -> dict:
        start = time.time()
        resp = self._openai.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=messages,
            temperature=config.TEMPERATURE,
            max_tokens=config.MAX_TOKENS,
        )
        latency_ms = int((time.time() - start) * 1000)
        return {
            "text": resp.choices[0].message.content,
            "latency_ms": latency_ms,
            "prompt_tokens": resp.usage.prompt_tokens,
            "completion_tokens": resp.usage.completion_tokens,
        }

    def _chat_oss(self, messages: list[dict]) -> dict:
        # Space endpoint takes a single string. Flatten messages into one prompt.
        prompt = self._flatten(messages)
        start = time.time()
        text = self._gradio.predict(prompt, api_name="/chat")
        latency_ms = int((time.time() - start) * 1000)
        # Space gives no token counts; approximate (~4 chars/token).
        prompt_tokens = sum(len(m["content"]) for m in messages) // 4
        completion_tokens = len(str(text)) // 4
        return {
            "text": str(text),
            "latency_ms": latency_ms,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
        }

    @staticmethod
    def _flatten(messages: list[dict]) -> str:
        lines = []
        for m in messages:
            role = m["role"].capitalize()
            lines.append(f"{role}: {m['content']}")
        lines.append("Assistant:")
        return "\n".join(lines)