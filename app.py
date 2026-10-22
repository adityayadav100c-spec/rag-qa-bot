import os
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).parent
DB_PATH = ROOT / "chroma_db"
COLLECTION = "northstar-demo"


def build_retriever():
    source = ROOT / "data" / "product_guide.txt"
    text = source.read_text(encoding="utf-8")
    chunks = RecursiveCharacterTextSplitter(
        chunk_size=450, chunk_overlap=60
    ).split_documents([Document(page_content=text, metadata={"source": source.name})])
    store = Chroma(
        collection_name=COLLECTION,
        persist_directory=str(DB_PATH),
        embedding_function=OpenAIEmbeddings(),
    )
    if store._collection.count() == 0:
        store.add_documents(chunks)
    return store.as_retriever(search_kwargs={"k": 3})


def answer(question: str) -> str:
    question = question.strip()
    if not question:
        return "Please enter a question."
    if not os.getenv("OPENAI_API_KEY"):
        return "Set OPENAI_API_KEY in your .env file before asking questions."

    docs = build_retriever().invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Answer using only the supplied context. If it does not contain "
                "the answer, say you do not know. Keep the answer concise.\n\n"
                "Context:\n{context}",
            ),
            ("human", "{question}"),
        ]
    )
    response = (prompt | ChatOpenAI(model="gpt-4o-mini", temperature=0)).invoke(
        {"context": context, "question": question}
    )
    sources = ", ".join(sorted({doc.metadata["source"] for doc in docs}))
    return f"{response.content}\n\nSources: {sources}"


load_dotenv(ROOT / ".env")
demo = gr.Interface(
    fn=answer,
    inputs=gr.Textbox(label="Ask about the fictional Northstar products"),
    outputs=gr.Markdown(),
    title="Northstar Product Q&A (RAG demo)",
    description="Answers are retrieved from the included sample product guide.",
)

if __name__ == "__main__":
    demo.launch()
