from pathlib import Path

import gradio as gr
from dotenv import load_dotenv

from rag import answer

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")

demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(
        label="Ask a question",
        placeholder="How do I return an item that arrived damaged?",
        lines=2,
    ),
    outputs=gr.Markdown(label="Answer and sources"),
    title="Northstar Help Center Q&A",
    description=(
        "A retrieval-augmented demo over the included fictional handbook. "
        "Try a sample question below; verify important answers against the sources."
    ),
    examples=[
        ["How do I wash the Travel Mug?"],
        ["Is the Daypack waterproof?"],
        ["How long do I have to return an unused product?"],
        ["What are your current store hours?"],
    ],
)

if __name__ == "__main__":
    demo.launch()
