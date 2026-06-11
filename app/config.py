import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI (frontier)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_JUDGE_MODEL = os.getenv("OPENAI_JUDGE_MODEL", "gpt-4o")

# OSS (Hugging Face Space)
HF_SPACE_URL = os.getenv("HF_SPACE_URL", "")
HF_TOKEN = os.getenv("HF_TOKEN", "")

# Assistant behavior
GUARDRAILS_ENABLED = os.getenv("GUARDRAILS_ENABLED", "true").lower() == "true"
MEMORY_WINDOW = int(os.getenv("MEMORY_WINDOW", "10"))

# Generation defaults
TEMPERATURE = 0.7
MAX_TOKENS = 512
# OSS reply cap, passed to the Space per request (was hardcoded 128 in the Space).
OSS_MAX_NEW_TOKENS = int(os.getenv("OSS_MAX_NEW_TOKENS", "128"))