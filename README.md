# Generative AI Q&A Bot (RAG)

A local document question-answering demo built with LangChain, Chroma, OpenAI
embeddings/chat, and Gradio. It chunks a fictional multi-section product help
center, indexes passages, retrieves relevant source text, and instructs the
language model to abstain when the supplied context does not contain an answer.

## Features

- Multi-document `.md`, `.txt`, and text-extractable `.pdf` ingestion with
  source/page metadata.
- 700-character chunks with 100-character overlap and bounded top-k retrieval.
- Persistent local Chroma vector store; changing source content creates a
  distinct collection rather than silently reusing a stale index.
- Grounded responses that display source filenames.
- Retrieved source sections listed deterministically alongside the generated
  answer, independent of whether the model formats its own citations.
- Retrieval-only evaluation against 12 questions, without using an LLM to grade
  itself.
- Gradio UI with examples and explicit API-key errors.
- Synthetic content covering products, order policies, shipping, returns,
  troubleshooting, support, privacy, and out-of-scope questions.

## Run

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
# Add your OpenAI API key to .env
python app.py
```

Open the local Gradio URL shown in the terminal. API usage may incur charges.
Never commit `.env`.

Run the retrieval smoke evaluation:

```powershell
python evaluate_retrieval.py
```

This prints Recall@1, Recall@3, and mean reciprocal rank using the labelled
question set. It measures retrieval only, not answer correctness or factuality.

## Notes

- `data/` contains explicitly fictional content, not real customer documents or
  real company policy.
- Place your own authorized, text-extractable PDF, Markdown, or text files in
  `data/`; scanned-image PDFs require OCR before this loader can read them.
- [`docs/EVALUATION.md`](./docs/EVALUATION.md) explains the retrieval metrics
  and what they do not measure.
- The vector store is generated locally in `chroma_db/` and ignored by Git.
- Answers can still be inaccurate or omit evidence. Inspect the sources and
  verify important claims.
- This starter does not implement authentication, multi-tenant isolation,
  document access control, PII redaction, or production monitoring.

Run document-ingestion and evaluation-set checks with
`python -m unittest discover -s tests`.

GitHub Actions runs dependency installation, Python compilation, and the test
suite on pushes and pull requests.
