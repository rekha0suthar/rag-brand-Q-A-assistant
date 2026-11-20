"""Build the FAISS index in `index_dir` from the files in ./docs.

Run as a script after changing docs/: `python ingest.py`. The chunking step
(chunk_texts) is also imported by the Streamlit upload mode so both index
builds use identical splitter settings.
"""

from pathlib import Path

from utils.config import embedding_model_name, load_config
from utils.io import load_docs

# Splitters: support both old/new LangChain
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS


def chunk_texts(docs, chunk_size=800, chunk_overlap=150) -> list[dict]:
    """
    Pure chunking step: (filename, text) pairs -> chunk dicts with source metadata.

    Shared by build_index() (disk index from ./docs) and the Streamlit
    "Your documents" mode (in-memory index from uploads) so both use
    identical splitter settings.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = []
    for fname, text in docs:
        pieces = splitter.split_text(text)
        for ch in pieces:
            ch = (ch or "").strip()
            if ch:
                chunks.append({"page_content": ch, "metadata": {"source": fname}})
    return chunks


def build_text_chunks(docs_dir="docs") -> list[dict]:
    """Load every readable file in docs_dir and chunk it with config.yaml settings."""
    cfg = load_config()
    docs = load_docs(docs_dir)
    if not docs:
        raise RuntimeError(
            "[ingest] No readable docs found. Put .md/.txt text files or text-based PDFs in ./docs"
        )
    chunks = chunk_texts(
        docs,
        chunk_size=cfg.get("chunk_size", 800),
        chunk_overlap=cfg.get("chunk_overlap", 150),
    )
    print(f"[ingest] Total chunks: {len(chunks)}")
    return chunks


def build_index():
    """Chunk ./docs, embed locally and write the FAISS index to index_dir."""
    cfg = load_config()
    index_dir = cfg.get("index_dir", "faiss_index")
    Path(index_dir).mkdir(parents=True, exist_ok=True)

    chunks = build_text_chunks("docs")
    texts = [c["page_content"] for c in chunks]
    metas = [c["metadata"] for c in chunks]

    # Must match the embedding model chain.py loads at query time -- using a
    # different one here would silently change the index's vector dimension
    # and break (or corrupt) retrieval. Both sides read cfg["embedding_model"].
    from langchain_huggingface import HuggingFaceEmbeddings

    model_name = embedding_model_name(cfg)
    print(f"[ingest] Using local HF embeddings: {model_name}")
    embeddings = HuggingFaceEmbeddings(model_name=model_name)

    print("[ingest] Building FAISS index…")
    vs = FAISS.from_texts(texts, embeddings, metadatas=metas)
    vs.save_local(index_dir)
    # Plain ASCII: Windows consoles default to cp1252 and crash on emoji.
    print(f"[ingest] OK - indexed {len(texts)} chunks into {index_dir}")


if __name__ == "__main__":
    build_index()
