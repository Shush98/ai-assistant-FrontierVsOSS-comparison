import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI (frontier)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")

# Judge / evaluator. Default to Anthropic (Claude) so the judge is from a DIFFERENT model
# family than the frontier model under test (GPT) — avoids same-family evaluator bias in
# both the live arena KPIs and the offline report metrics. Flip JUDGE_PROVIDER=openai to
# compare judges apples-to-apples. Used by app/judge_client.py.
JUDGE_PROVIDER = os.getenv("JUDGE_PROVIDER", "anthropic").lower()  # "anthropic" | "openai"
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_JUDGE_MODEL = os.getenv("ANTHROPIC_JUDGE_MODEL", "claude-sonnet-4-6")

# OSS (Hugging Face Space) — all OSS-path config lives here.
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")
# OSS reply cap, passed to the Space per request (was hardcoded 128 in the Space).
OSS_MAX_NEW_TOKENS = int(os.getenv("OSS_MAX_NEW_TOKENS", "128"))

# Assistant behavior
GUARDRAILS_ENABLED = os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true"
MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "10"))

# Generation defaults
TEMPERATURE = 0.7
MAX_TOKENS = 512