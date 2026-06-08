import time
from openai import OpenAI
from app import config


class LLMClient:
    """One interface, two providers: 'frontier' (OpenAI) or 'oss' (HF Space)."""

    def __init__(self, provider: str):
        if provider not in ("frontier", "oss"):
            raise ValueError(f"unknown provider: {provider}")
        self.provider = provider
        if provider == "frontier":
            self._openai = OpenAI(api_key=config.OPENAI_API_KEY)

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
        # TODO: wire to HF Space after deploy (Step ~9)
        raise NotImplementedError("OSS provider not wired yet — HF Space pending")