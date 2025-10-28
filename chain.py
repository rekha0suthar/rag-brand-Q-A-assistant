# chain_ollama.py  (use this as chain.py if going local)
import time, yaml
from pathlib import Path
from langchain_community.vectorstores import FAISS
from utils.logger import pack_result

try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate
try:
    from langchain_core.messages import HumanMessage, SystemMessage
except ImportError:
    from langchain.schema import HumanMessage, SystemMessage

def _hf_embeddings(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name=model_name)

def _ollama_chat(model="llama3"):
    from langchain_community.chat_models import ChatOllama
    return ChatOllama(model=model, temperature=0)

PROMPT = PromptTemplate.from_template(
"""You are a brand-safe marketing assistant.
Use ONLY the provided CONTEXT. If the answer is not present, reply exactly: {refusal_text}

Respond concisely in an on-brand tone. Cite sources as [source: <filename>].

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""
)

def get_cfg():
    return yaml.safe_load(Path("config.yaml").read_text())

def load_chain():
    cfg = get_cfg()
    emb = _hf_embeddings("sentence-transformers/all-MiniLM-L6-v2")
    vs = FAISS.load_local(cfg["index_dir"], embeddings=emb, allow_dangerous_deserialization=True)
    llm = _ollama_chat("llama3")
    print("[chain] Using local Ollama: llama3")

    def ask(question: str):
        t0 = time.time()
        docs = vs.similarity_search(question, k=cfg["top_k"])
        ctx = "\n\n".join(d.page_content for d in docs)
        sources = [d.metadata.get("source", "unknown") for d in docs]
        if len(ctx) < cfg["min_context_chars"]:
            return pack_result(cfg["refusal_text"], sources, time.time() - t0)
        prompt_text = PROMPT.format(context=ctx, question=question, refusal_text=cfg["refusal_text"])
        messages = [
            SystemMessage(content="You are a careful, brand-safe assistant that never invents facts."),
            HumanMessage(content=prompt_text),
        ]
        out = llm.invoke(messages)
        ans = getattr(out, "content", out)
        return pack_result(ans, sources, time.time() - t0)

    return ask
