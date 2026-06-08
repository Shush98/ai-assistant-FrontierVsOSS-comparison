import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype="auto")


def chat(message):
    # Stateless: backend packs all context into `message`. Space holds NO memory.
    messages = [{"role": "user", "content": message}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer([text], return_tensors="pt")
    generated = model.generate(**inputs, max_new_tokens=512, temperature=0.7, do_sample=True)
    output_ids = generated[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(output_ids, skip_special_tokens=True)


demo = gr.Interface(fn=chat, inputs="text", outputs="json", api_name="chat")

if __name__ == "__main__":
    demo.launch()