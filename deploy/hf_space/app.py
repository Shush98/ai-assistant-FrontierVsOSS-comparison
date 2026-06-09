import json
import time

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto")


def chat(messages_json, tools_json):
    """STATELESS native tool-calling endpoint.

    The backend sends the full conversation as `messages_json` (a JSON list of
    {role, content, ...} dicts) plus `tools_json` (JSON list of tool schemas).
    We run Qwen2.5's trained tool template and return {"text", "server_ms"}:
    `text` is the RAW generated text (may contain a <tool_call>{...}</tool_call>
    block); `server_ms` is the TRUE inference latency timed inside the Space, so
    the backend can separate model compute from gradio_client/network overhead.
    The Space does NOT run tools and holds NO state; the backend owns the loop.
    """
    messages = json.loads(messages_json) if messages_json else []
    tools = json.loads(tools_json) if tools_json else None

    t0 = time.time()
    text = tokenizer.apply_chat_template(
        messages,
        tools=tools,
        add_generation_prompt=True,
        tokenize=False,
    )
    inputs = tokenizer([text], return_tensors="pt")
    # max_new_tokens bounds CPU generation time (the dominant latency on a free
    # CPU Space): 128 comfortably covers normal chat replies, and the model still
    # stops early at EOS for short answers.
    generated = model.generate(
        **inputs, max_new_tokens=128, temperature=0.7, do_sample=True
    )
    output_ids = generated[0][len(inputs.input_ids[0]):]
    out = tokenizer.decode(output_ids, skip_special_tokens=True)
    server_ms = int((time.time() - t0) * 1000)  # true inference latency
    return {"text": out, "server_ms": server_ms}


demo = gr.Interface(
    fn=chat,
    inputs=["text", "text"],   # messages_json, tools_json
    outputs="json",            # returns {text, server_ms}
    api_name="chat",
)

if __name__ == "__main__":
    demo.launch()
