# FastAPI Docker Space (free CPU tier). HF runs the container with the README's
# `app_port` (7860) exposed.
FROM python:3.11-slim

# Build tools: fallback so llama-cpp-python can compile from source if no
# prebuilt CPU wheel matches (the extra index below usually provides one).
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git \
    && rm -rf /var/lib/apt/lists/*

# HF Docker Spaces run as uid 1000; create that user and keep everything under
# its HOME so the hub cache and app dir are writable.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt \
    --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu

# Bake the model + tokenizer into the image so a cold start (Space waking from
# sleep) does NOT re-download ~400MB — it just loads from the local cache.
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen2.5-0.5B-Instruct-GGUF', allow_patterns=['*q4_k_m.gguf'])"
RUN python -c "from transformers import AutoTokenizer; \
    AutoTokenizer.from_pretrained('Qwen/Qwen2.5-0.5B-Instruct')"

COPY --chown=user app.py .

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
