import gradio as gr
from transformers import AutoModelForCausalLM, pipeline
from peft import PeftModel

def load_model():
    base_model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2b-it",
        device_map={"": "cpu"}
    )
    model = PeftModel.from_pretrained(base_model, "./gemma2-finetuned")
    return pipeline(
        "text-generation",
        model=model,
        device=-1,
        tokenizer="google/gemma-2b-it"
    )

generator = load_model()

def generate_response(input_text):
    prompt = f"Generate meaning representation: {input_text}\nOutput:"
    result = generator(
        prompt,
        max_new_tokens=100,
        do_sample=False
    )
    return result[0]['generated_text'].split("Output:")[-1].strip()

gr.Interface(
    fn=generate_response,
    inputs=gr.Textbox(label="Input Sentence", lines=2),
    outputs=gr.Textbox(label="Structured Output", lines=3),
    title="Gemma 2B Meaning Representation Generator",
    examples=[
        ["Does this game support multiplayer?"],
        ["What's the release year of Final Fantasy VII?"],
        ["Is this title available on Steam?"]
    ]
).launch()