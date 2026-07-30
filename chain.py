# chain.py
import os, time, yaml
from pathlib import Path

# LangChain imports with 0.2-compatible fallbacks
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate
try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    from langchain.schema import HumanMessage, SystemMessage

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

from utils.logger import pack_result

print("[chain] module imported OK")  # sentinel to prove import succeeded

PROMPT = PromptTemplate.from_template(
"""You are a brand-safe marketing assistant.
Use ONLY the provided CONTEXT. If the answer is not present, reply exactly: {refusal_text}

Respond concisely in an on-brand tone. Cite sources as [source: <filename>].

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
)

def _get_cfg():
    cfg_path = Path("config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError("config.yaml not found in project root.")
    return yaml.safe_load(cfg_path.read_text())

def load_chain():
    """
    Factory that initializes retrieval + LLM and returns a callable: ask(question) -> dict.
    Kept inside a function so importing chain.py never triggers heavy work.
    """
    cfg = _get_cfg()

    # 1) Load FAISS with the SAME embeddings used at ingest
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.load_local(
        cfg.get("index_dir", "faiss_index"),
        embeddings=emb,
        allow_dangerous_deserialization=True
    )

    # 2) Initialize the LLM. LLM_PROVIDER env var overrides config.yaml so a
    # deploy host (e.g. Render, set LLM_PROVIDER=groq) doesn't require
    # editing the file -- "groq" needs no locally-running server, "ollama"
    # is for local dev where `ollama serve` is actually running.
    provider = os.getenv("LLM_PROVIDER", cfg.get("provider", "ollama"))
    temperature = cfg.get("temperature", 0.0)

    if provider == "groq":
        from langchain_groq import ChatGroq
        model_name = cfg.get("groq_model", "llama-3.3-70b-versatile")
        llm = ChatGroq(model=model_name, temperature=temperature)
    else:
        from langchain_ollama import ChatOllama
        model_name = cfg.get("ollama_model", "llama3")
        llm = ChatOllama(model=model_name, temperature=temperature)

    print(f"[chain] Using {provider} model: {model_name}")

    def ask(question: str):
        t0 = time.time()
        docs = vs.similarity_search(question, k=cfg.get("top_k", 4))
        ctx = "\n\n".join(d.page_content for d in docs)
        sources = [d.metadata.get("source", "unknown") for d in docs]
        if len(ctx) < cfg.get("min_context_chars", 200):
            return pack_result(cfg.get("refusal_text", "Not enough info in the docs."), sources, time.time() - t0)
        prompt_text = PROMPT.format(
            context=ctx,
            question=question,
            refusal_text=cfg.get("refusal_text", "Not enough info in the docs.")
        )
        messages = [
            SystemMessage(content="You are a careful, brand-safe assistant that never invents facts."),
            HumanMessage(content=prompt_text),
        ]
        out = llm.invoke(messages)
        ans = getattr(out, "content", out)
        return pack_result(ans, sources, time.time() - t0)

    return ask
