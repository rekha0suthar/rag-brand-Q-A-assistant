# chain.py
import os, time, yaml
from pathlib import Path

# Load .env before any os.getenv call -- otherwise local Groq runs
# (LLM_PROVIDER=groq) fail auth unless the key is exported manually.
from dotenv import load_dotenv
load_dotenv()

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

from utils.errors import QuotaExceededError, is_quota_error
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

def filter_by_score(results, threshold, margin=None):
    """
    Keep only (doc, score) pairs whose score passes the threshold.

    FAISS (as built by ingest.py via FAISS.from_texts) returns L2 DISTANCE,
    so LOWER is better: a doc passes when score <= threshold.

    If `margin` is given (relative-margin rule, used by the session-upload
    gate): when the BEST chunk passes the threshold, also keep chunks within
    `margin` of the best score even if they sit above the threshold -- on an
    arbitrary uploaded corpus the absolute distances shift, but chunks packed
    close to the best match are usually part of the same answer. If even the
    best chunk is above the threshold, nothing is kept (refuse).
    """
    if not results:
        return []
    best = min(s for _, s in results)
    if best > threshold:
        return []
    cutoff = threshold if margin is None else max(threshold, best + margin)
    return [(d, s) for d, s in results if s <= cutoff]

def _get_cfg():
    cfg_path = Path("config.yaml")
    if not cfg_path.exists():
        raise FileNotFoundError("config.yaml not found in project root.")
    return yaml.safe_load(cfg_path.read_text())

def build_llm(cfg):
    """
    Initialize the LLM. LLM_PROVIDER env var overrides config.yaml so a
    deploy host (e.g. Render, set LLM_PROVIDER=groq) doesn't require
    editing the file -- "groq" needs no locally-running server, "ollama"
    is for local dev where `ollama serve` is actually running.
    """
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
    return llm

def make_asker(vectorstore, cfg, llm, score_threshold=None, score_margin=None):
    """
    Build the ask(question) -> dict callable over ANY vectorstore.

    Single code path for retrieval + threshold gate + grounded generation:
    load_chain() wraps it with the committed disk index (demo mode, API,
    eval harness), and the Streamlit "Your documents" mode wraps it with an
    in-memory index built from uploads. Keeping it here means the refusal
    gate can never drift between entry points.

    Gate parameters (both optional, so demo-mode callers are unchanged):
      score_threshold -- overrides cfg["score_threshold"]. Session mode passes
        cfg["session_score_threshold"]: natural questions over arbitrary
        uploaded documents score systematically higher L2 distances than the
        demo corpus the default threshold was calibrated on.
      score_margin -- enables the relative-margin rule in filter_by_score.
    """
    threshold = score_threshold if score_threshold is not None else cfg.get("score_threshold", 1.25)

    def ask(question: str):
        t0 = time.time()
        results = vectorstore.similarity_search_with_score(question, k=cfg.get("top_k", 4))
        # Deterministic refusal gate: drop chunks whose L2 distance is above
        # the calibrated threshold. If nothing is close enough, refuse WITHOUT
        # calling the LLM and WITHOUT citing sources -- a refusal that cites
        # documents it didn't use would be misleading.
        kept = filter_by_score(results, threshold, margin=score_margin)
        if not kept:
            best = round(min((float(s) for _, s in results), default=0.0), 3) if results else None
            return pack_result(
                cfg.get("refusal_text", "Not enough info in the docs."), [], time.time() - t0,
                extra={"best_score": best, "threshold": threshold},
            )
        docs = [d for d, _ in kept]
        ctx = "\n\n".join(d.page_content for d in docs)
        sources = [d.metadata.get("source", "unknown") for d in docs]
        prompt_text = PROMPT.format(
            context=ctx,
            question=question,
            refusal_text=cfg.get("refusal_text", "Not enough info in the docs.")
        )
        messages = [
            SystemMessage(content="You are a careful, brand-safe assistant that never invents facts."),
            HumanMessage(content=prompt_text),
        ]
        try:
            out = llm.invoke(messages)
        except Exception as e:
            # Map provider rate/quota failures to a clean, user-facing error;
            # everything else propagates for the caller to log.
            if is_quota_error(e):
                raise QuotaExceededError() from e
            raise
        ans = getattr(out, "content", out)
        return pack_result(ans, sources, time.time() - t0)

    return ask

def load_chain():
    """
    Factory that initializes retrieval + LLM and returns a callable: ask(question) -> dict.
    Kept inside a function so importing chain.py never triggers heavy work.
    """
    cfg = _get_cfg()

    # Load FAISS with the SAME embeddings used at ingest
    emb = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.load_local(
        cfg.get("index_dir", "faiss_index"),
        embeddings=emb,
        allow_dangerous_deserialization=True
    )

    llm = build_llm(cfg)
    return make_asker(vs, cfg, llm)
