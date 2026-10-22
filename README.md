# Generative AI Q&A Bot (RAG)

A small retrieval-augmented question-answering demo built with LangChain, Chroma, OpenAI embeddings/chat, and Gradio. It indexes the included fictional product guide and retrieves relevant passages before answering.

## Run

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Add your OpenAI API key to .env
python app.py
```

Open the local Gradio URL shown in the terminal. API use may incur charges. Do not commit `.env`.

## Notes

- `data/` contains synthetic demo content, not real customer documents.
- The vector store is generated locally in `chroma_db/` and is ignored by Git.
- Answers are grounded in retrieved text, but can still be inaccurate; verify important claims.
