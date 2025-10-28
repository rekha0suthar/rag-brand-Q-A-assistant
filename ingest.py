# ingest.py
import yaml
from dotenv import load_dotenv
from pathlib import Path
from typing import List
from utils.io import load_docs

# Splitters: support both old/new LangChain
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS

# Prefer OpenAI if quota is available; else fall back to local HF
USE_OPENAI_DEFAULT = True

def get_cfg():
    return yaml.safe_load(Path("config.yaml").read_text())

def build_text_chunks(docs_dir="docs") -> List[dict]:
    cfg = get_cfg()
    docs = load_docs(docs_dir)
    if not docs:
        raise RuntimeError(
            "[ingest] No readable docs found. Put .md/.txt text files or text-based PDFs in ./docs"
        )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cfg.get("chunk_size", 800),
        chunk_overlap=cfg.get("chunk_overlap", 150),
    )
    chunks = []
    for fname, text in docs:
        pieces = splitter.split_text(text)
        for ch in pieces:
            ch = (ch or "").strip()
            if ch:
                chunks.append({"page_content": ch, "metadata": {"source": fname}})
    print(f"[ingest] Total chunks: {len(chunks)}")
    return chunks

def _openai_embeddings(model_name: str):
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model=model_name)

def _hf_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name)

def build_index():
    load_dotenv()
    cfg = get_cfg()
    index_dir = cfg.get("index_dir", "faiss_index")
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    chunks = build_text_chunks("docs")
    texts = [c["page_content"] for c in chunks]
    metas = [c["metadata"] for c in chunks]

    embeddings = None
    if USE_OPENAI_DEFAULT:
        try:
            model = cfg.get("embedding_model", "text-embedding-3-small")
            print(f"[ingest] Trying OpenAI embeddings: {model}")
            embeddings = _openai_embeddings(model)
            # warm-up small call to fail fast if quota is exhausted
            _ = embeddings.embed_documents(texts[:2])
        except Exception as e:
            print(f"[ingest] OpenAI embeddings unavailable ({e.__class__.__name__}). Falling back to HF.")
            embeddings = None

    if embeddings is None:
        print("[ingest] Using local HF embeddings: sentence-transformers/all-MiniLM-L6-v2")
        embeddings = _hf_embeddings("sentence-transformers/all-MiniLM-L6-v2")

    print("[ingest] Building FAISS index…")
    vs = FAISS.from_texts(texts, embeddings, metadatas=metas)
    vs.save_local(index_dir)
    print(f"[ingest] ✅ Indexed {len(texts)} chunks into {index_dir}")

if __name__ == "__main__":
    build_index()
